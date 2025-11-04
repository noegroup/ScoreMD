from hydra_zen import builds, just

from scoremd.data.augment import apply_random_rotations
from scoremd.data.dataset.aldp import ALDPDataset, CoarseGrainingLevel
from scoremd.data.dataset.minipeptide import CGMinipeptideDataset
from scoremd.data.dataset.mueller import MuellerBrownSimulation
from scoremd.data.dataset.protein import SingleProteinDataset
from scoremd.data.dataset.toy import ToyDataset, ToyDatasets
import scoremd.diffusion.classic.utils as diffusion_utils

# import straight line diffusion loss
from scoremd.loss import RangedLoss
from scoremd.training import equal_weight, exponential_decay
from scoremd.training.optimizer import get_constant_lr_optimizer, get_cosine_lr_optimizer
from scoremd.training.schedule import IDENTITY, OneAfterAnother
from scoremd.training.weighting import (
    construct_global_constant_weighting_function,
    construct_ranged_constant_weighting_function,
)
from scoremd.utils.file import get_persistent_storage


def create_dataset_store(store):
    base_dir = get_persistent_storage()

    dataset_store = store(group="dataset")

    ToyDatasetBuilder = builds(ToyDataset, n_samples=10_000, populate_full_signature=True)
    dataset_store(ToyDatasetBuilder(example=ToyDatasets.CheckerBoard, name="checker_board"), name="checker_board")
    dataset_store(ToyDatasetBuilder(example=ToyDatasets.DoubleWell, name="double_well"), name="double_well")
    dataset_store(ToyDatasetBuilder(example=ToyDatasets.DoubleWell2D, name="double_well_2d"), name="double_well_2d")
    dataset_store(
        builds(
            MuellerBrownSimulation,
            n_samples=100_000,
            kbT=23.0,
            mass=(0.5, 0.5),
            gamma=1.0,
            n_steps=50,
            dt=5e-3,
            populate_full_signature=True,
        ),
        name="mueller_brown",
    )
    dataset_store(builds(ALDPDataset, name="aldp", populate_full_signature=True), name="aldp")
    dataset_store(
        builds(ALDPDataset, limit_samples=1, validation=False, name="aldp_single", populate_full_signature=True),
        name="aldp_single",
    )

    dataset_store(
        builds(
            ALDPDataset,
            name="aldp_six",
            path=f"{base_dir}/aldp/CG_300k_100kx1ps.npy",
            coarse_graining_level=CoarseGrainingLevel.SIX_BEADS,
            validation=False,
            populate_full_signature=True,
        ),
        name="aldp_six",
    )

    dataset_store(
        builds(
            CGMinipeptideDataset,
            pdb_directory=f"{base_dir}/minipeptides/pdbs",
            train_path=f"{base_dir}/minipeptides/train.npy",
            val_path=f"{base_dir}/minipeptides/val.npy",
            # test_path=f"{base_dir}/minipeptides/test.npy",  # intentionally left out, so that normally it evaluates on validation set
            name="minipeptides",
            populate_full_signature=True,
        ),
        name="minipeptides",
    )

    def generate_protein_dataset(name, file_names, tica_file_name, topology_file_name):
        dataset_store(
            builds(
                SingleProteinDataset,
                paths=[f"{base_dir}/deshaw/{file_name}" for file_name in file_names],
                tica_path=f"{base_dir}/deshaw/{tica_file_name}",
                topology_path=f"{base_dir}/deshaw/{topology_file_name}",
                name=name,
                populate_full_signature=True,
            ),
            name=name,
        )

    generate_protein_dataset("chignolin", ["chignolin-0_ca.h5"], "chignolin_tica.pic", "chignolin.pdb")
    generate_protein_dataset("trpcage", ["trpcage-0_ca.h5"], "trpcage_tica.pic", "trpcage.pdb")
    generate_protein_dataset("bba", ["bba-0_ca.h5", "bba-1_ca.h5"], "bba_tica.pic", "bba.pdb")

    return dataset_store


def create_weighting_store(store):
    weighting_store = store(group="weighting_function")
    weighting_store(builds(construct_global_constant_weighting_function, zen_partial=True), name="constant")
    weighting_store(
        builds(construct_ranged_constant_weighting_function, normalize=True, zen_partial=True), name="ranged"
    )

    return weighting_store


def create_optimizer_store(store):
    optimizer_store = store(group="optimizer")
    optimizer_store(
        builds(get_constant_lr_optimizer, learning_rate=3e-4, clip=1e3, zen_partial=True, populate_full_signature=True),
        name="constant",
    )

    optimizer_store(
        builds(
            get_cosine_lr_optimizer,
            learning_rate=3e-4,
            min_learning_rate=1e-5,
            clip=1e3,
            zen_partial=True,
            populate_full_signature=True,
        ),
        name="cosine",
    )

    return optimizer_store


def create_trainig_schedule_store(store):
    training_schedule_store = store(group="training_schedule")

    standard_loss = builds(
        diffusion_utils.get_loss,
        likelihood_weighting=False,
        sliced=False,
        div_est=1,
        gaussian_div_est=False,
        partial_t_approx=False,
        zen_partial=True,
        populate_full_signature=True,
    )

    augment_store = store(group="training_schedule/augment")
    no_augmentation = just(IDENTITY)
    augment_store(no_augmentation, name="no_augmentation")
    augment_store(just(apply_random_rotations), name="random_rotations")

    loss_store = store(group="training_schedule/losses")

    eps = 1e-5
    single_vp = just(
        [
            builds(
                RangedLoss,
                loss=standard_loss,
                range=[1.0, eps],
                time_weighting=builds(
                    equal_weight,
                    midpoint=0.3,
                    zen_partial=True,
                    populate_full_signature=True,
                ),
                populate_full_signature=True,
            )
        ]
    )
    dual_vp = just([single_vp[0], single_vp[0]])

    loss_store(single_vp, name="single_vp")
    loss_store(dual_vp, name="dual_vp")

    training_schedule_store(
        builds(
            OneAfterAnother,
            losses=single_vp,
            BS=128,
            BS_factor=1,
            epochs=[200],
            augment=no_augmentation,
            populate_full_signature=True,
        ),
        name="vp_standard",
    )

    training_schedule_store(
        builds(
            OneAfterAnother,
            losses=dual_vp,
            BS=128,
            BS_factor=1,
            epochs=[200, 20],
            augment=no_augmentation,
            populate_full_signature=True,
        ),
        name="vp_finetune",
    )

    def build_three_model_store(last_time_weighting):
        return builds(
            OneAfterAnother,
            losses=just(
                [
                    builds(
                        RangedLoss,
                        loss=standard_loss,
                        range=[1.0, 0.6],
                        time_weighting=builds(
                            equal_weight,
                            midpoint=0.8,
                            t0=0.6,
                            t1=1.0,
                            zen_partial=True,
                            populate_full_signature=True,
                        ),
                        trainable=[0],
                        evaluated_models=[0],
                        populate_full_signature=True,
                    ),
                    builds(
                        RangedLoss,
                        loss=standard_loss,
                        range=[0.6, 0.1 - 0.001],
                        time_weighting=builds(
                            equal_weight,
                            midpoint=0.35,
                            t0=0.1,
                            t1=0.6,
                            zen_partial=True,
                            populate_full_signature=True,
                        ),
                        trainable=[1],
                        evaluated_models=[0, 1],
                        populate_full_signature=True,
                    ),
                    builds(
                        RangedLoss,
                        loss=standard_loss,
                        # we subtract a small epsilon, to make FP loss more stable
                        range=[0.1 - 0.001, eps],
                        time_weighting=last_time_weighting,
                        trainable=[2],
                        evaluated_models=[1, 2],
                        populate_full_signature=True,
                    ),
                ],
            ),
            BS=128,
            BS_factor=1,
            epochs=[200, 200, 200],
            augment=no_augmentation,
            populate_full_signature=True,
        )

    training_schedule_store(
        build_three_model_store(
            builds(
                equal_weight,
                midpoint=0.03,
                t0=0.0,
                t1=0.1,
                zen_partial=True,
                populate_full_signature=True,
            )
        ),
        name="vp_three_models",
    )

    training_schedule_store(
        build_three_model_store(
            builds(
                exponential_decay,
                n=2,
                t0=0.0,
                t1=0.1,
                zen_partial=True,
                populate_full_signature=True,
            )
        ),
        name="vp_three_models_exp",
    )
