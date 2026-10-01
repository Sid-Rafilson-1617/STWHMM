import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from stwhmm_jax_optimized import STWHMM


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--save-dir", required=True)

    p.add_argument("--r", type=float, default=0.8)

    p.add_argument("--nStates", type=int, required=True)
    p.add_argument("--nTaus", type=int, required=True)

    p.add_argument("--restart", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)

    p.add_argument("--Niters", type=int, default=10)
    p.add_argument("--eps", type=float, default=1e-9)

    p.add_argument("--plot", type=bool, default=False)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()



def main(args):

    # set the random seed for reproducibility
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    # define the current save directory for this restart and transition matrix size
    current_save_dir = os.path.join(args.save_dir, f"N_{args.nStates}", f"restart_{args.restart}")
    os.makedirs(current_save_dir, exist_ok=True)

    # load the data
    data = np.load(args.data)

    # set random walk with jumps for H_mask
    H_mask = np.zeros((args.nTaus, args.nTaus))
    for i in range(args.nTaus):
        for j in range(args.nTaus):
            if i != j:
                H_mask[i, j] = 1
    H_mask[0,:] = 1
    H_mask[:,0] = 1




    # define the model and set the observations
    model = STWHMM(
        M = args.nStates,
        S = args.nTaus,
        s = None,
        sequence_lengths = None,
        T_mask = None,
        H_mask = H_mask,
        random_state = rng,
        min_rate = 1e-6
    )

    results = model.fit(
        Y=data['observations'],
        n_iter = args.Niters,
        verbose = False,
        check = False,
        update_pi = True
    )

    training_likelihoods = results.log_likelihoods


    # plot the EM convergence
    if args.plot:
        plt.figure()
        plt.plot(np.asarray(training_likelihoods))
        plt.title(f"EM Convergence")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log-Likelihood")
        plt.tight_layout()
        plt.savefig(os.path.join(current_save_dir, f"em_convergence.png"))
        plt.close()

        # plot the transition matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array(model.T_), cmap="Blues", cbar_kws={"label": "Transition Probability (%)"}, vmin=0, vmax=1)
        plt.title("Learned Transition Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.tight_layout()
        plt.savefig(os.path.join(current_save_dir, f"learned_transition_matrix.png"))
        plt.close()



    # save the A and B matrices and the log likelihoods to the current save directory
    np.savez(
        os.path.join(current_save_dir, "fitted_model.npz"),
        T = model.T_,
        H = model.H_,
        Lambda = model.Lambda_,
        pi = model.pi_,
        training_likelihoods = training_likelihoods,
        nStates = args.nStates,
        restart = args.restart,
        seed = args.seed
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--nStates", type=int, required=True)
    parser.add_argument("--restart", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--Niters", type=int, required=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--nTaus", type=int, required=True)

    args = parser.parse_args()
    main(args)
