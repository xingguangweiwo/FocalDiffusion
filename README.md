# FocalDiffusion

This repository contains the reference implementation of **FocalDiffusion**.
The current codebase is organised around a training pipeline that adapts a
pre-trained SD3.5 diffusion backbone to focal stack inputs.

## Configuration files

To keep things simple the repository now ships a **single base configuration**
plus three tiny dataset presets:

- [`configs/base.yaml`](configs/base.yaml) contains every default hyper-
  parameter in one place (model, optimiser, logging, etc.).
- [`configs/hypersim.yaml`](configs/hypersim.yaml),
  [`configs/virtual_kitti.yaml`](configs/virtual_kitti.yaml) and
  [`configs/mixed.yaml`](configs/mixed.yaml) extend the base file via
  `defaults: [base]` and override just the dataset-specific paths and, when
  needed, batch size/accumulation.

You can point the training script at any of these files or create your own
variant by copying `configs/base.yaml` and editing the fields directly. The most
important keys to review are:

- `model.base_model_id`: Hugging Face id of the Stable Diffusion 3.5 checkpoint
  you have access to.
- `data.data_root`: absolute path that points to the root of your focal stack
  dataset (HyperSim, Virtual KITTI or a mixture).
- `data.*_filelist`: text files that enumerate the relative paths for each
  sample. Example file lists are provided under
  [`data/filelists/`](data/filelists/) and follow the format
  `<stack_directory>,<depth_map>,<num_images>`.
- `training.batch_size`, `training.gradient_accumulation_steps`: adapt these to
  your GPU memory budget.
- `logging.use_wandb`: set to `true` if you want accelerator to report metrics
  to Weights & Biases.

## Preparing datasets

1. Generate focal stacks from HyperSim or Virtual KITTI using your preferred
   point-spread-function simulator. Store the rendered stacks and the matching
   depth maps in a directory structure under `data_root`.
2. Create text file lists for each split. You can start from the templates in
   [`data/filelists`](data/filelists) and replace the placeholder entries with
   the relative paths for your data. Each line should describe a single sample.
3. Update the dataset section of the YAML configuration to reference your new
   file lists and the correct focal range for the simulated camera parameters.

## Launching training

Once the configuration files have been edited, launch training with:

```bash
python -m script.train --config configs/hypersim.yaml
```

Add `--debug` for verbose logging or `--dry-run` to validate the configuration
without starting the optimisation loop. Training requires the datasets to be in
place and the pre-trained SD3.5 weights to be accessible through Hugging Face.

> **Tip:** `script/train.py` resolves relative paths inside the YAML files with
> respect to the file that defined them. This means you can keep project-relative
> paths (for instance the sample file lists in [`data/filelists`](data/filelists))
> and only override `data_root` to point at your local dataset directory.

### Python dependencies

Running the training or inference scripts in full still requires the deep
learning stack listed in `environment.yaml` (PyTorch, diffusers, accelerate,
etc.).  The lightweight configuration tooling introduced in this change does
not depend on those packages; the command-line interface will fall back to a
pure Python YAML parser when PyYAML is absent so that `--dry-run` checks can be
performed in constrained environments.  When you are ready to train, make sure
to install the complete requirements and log in to Hugging Face so that the
Stable Diffusion 3.5 checkpoints can be fetched.

## Syncing the repository with GitHub

The code in this workspace lives in a local Git checkout. To publish it to your
own GitHub repository **and** download it again on another machine (such as your
PyCharm workstation), follow the sequence below.

### 1. Publish the workspace commits to GitHub

1. **Check the branch and commits**
   ```bash
   git status
   git log --oneline | head
   ```
   Confirm that you are on the branch you want to share (for example `work` or
   `main`) and that the latest commit message matches what you expect.
2. **Create the branch if it only exists locally**. If `git branch` lists only
   `main` but you would like to push a `work` branch, create it and stay on it:
   ```bash
   git checkout -b work   # skip if you are happy to push main
   ```
3. **Configure the remote once**
   ```bash
   git remote add origin https://github.com/<your-account>/FocalDiffusion.git
   ```
   Replace `<your-account>` with your GitHub user or organisation name. If the
   remote already exists you will see `remote origin already exists`; this is
   expected and you can continue.
4. **Push the current branch**
   ```bash
   git push -u origin HEAD
   ```
   Git will create the matching branch on GitHub (`work` if you created it in
   step 2, otherwise `main`) and upload every commit from this workspace. Use
   `git push -u origin main` if you explicitly want to update the `main` branch.
5. **Recover from network errors**. If the push fails with `Connection was
   reset`, switch the remote to SSH (which is more resilient on some networks)
   and push again:
   ```bash
   git remote set-url origin git@github.com:<your-account>/FocalDiffusion.git
   git push -u origin HEAD
   ```
   Alternatively retry after increasing the HTTP buffer size via
   `git config --global http.postBuffer 524288000`.

6. **Resolve `Permission denied (publickey)` errors**. When you switch to SSH
   for the first time, GitHub will only accept the connection if the machine
   has an SSH key registered with your account. Generate one (only once per
   machine) and copy the public part to GitHub ▸ *Settings* ▸ *SSH and GPG
   keys*:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"  # accept the defaults
   cat ~/.ssh/id_ed25519.pub                          # copy this output
   ```
   After saving the key on GitHub, verify the connection:
   ```bash
   ssh -T git@github.com
   ```
   If you prefer to keep using HTTPS instead of SSH, create a
   [personal access token](https://github.com/settings/tokens/new?scopes=repo)
   and run `git push` with the token as the password when prompted. Git will
   remember it if you enable the credential helper via
   `git config --global credential.helper store`.

Once `git push` finishes successfully, refresh the repository page on GitHub –
you should now see the same commit hash as in `git log`.

### 2. Update your other machines (PyCharm, laptops, …)

With the commits available on GitHub, pull them down on each additional
environment:

```bash
git pull origin work   # or 'git pull origin main' if that is the branch you pushed
```

If you do not have any local modifications and simply want to replace the
working tree with the remote state, you can run:

```bash
git fetch origin
git reset --hard origin/work   # adapt the branch name as needed
```

PyCharm exposes the same actions through *Git ▸ Fetch* followed by *Git ▸ Pull*.
After the pull completes, your local files will match the code that was pushed
from this workspace.

If you prefer to keep a different branch layout, adjust the commands above to
match your naming. The important step is to `git push` the commits from this
environment to the remote repository; until that happens the GitHub UI will
still display the previous state of the codebase.

### Quick smoke test

Before starting a full training run you can confirm that the repository is in a
healthy state without heavy dependencies:

```bash
python -m script.train --config configs/base.yaml --dry-run
python -m compileall src
```

The first command validates YAML parsing and directory creation; the second one
byte-compiles all modules to catch syntax errors early. Both commands run with
only the Python standard library installed. For real training you still need
the full deep-learning dependency stack described above.
