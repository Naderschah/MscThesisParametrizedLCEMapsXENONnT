{ pkgs ? import (fetchTarball {
  url = "https://github.com/NixOS/nixpkgs/archive/nixpkgs-24.11-darwin.tar.gz";
}) {} }:

let
  lib = pkgs.lib;
  # Define the Python environment with the required packages
  # TODO Change to 311 and incldue tensorflow
  pythonEnv = pkgs.python311.withPackages (pythonPackages: with pythonPackages; [
    #jupyterlab
    notebook
    pandas
    tensorflow
    matplotlib
    #h5py
  ]);
  inherit (pythonEnv) python;
  pythonPath = "${pythonEnv}/bin/python3";

  # Get Sha 
  # nix-prefetch-url --unpack https://github.com/MilesCranmer/DynamicDiff.jl/archive/refs/tags/v0.2.0.tar.gz
  # nix hash to-sri --type sha256 15is3rdsj0a0h4flj6lcf9xfj5knjbq8dfqrsgh2mjhwd5klrhap
  juliaEnvModBuildIn = pkgs.julia_110.overrideAttrs (oldAttrs: {
    buildInputs = ((lib.filter (input: input != pkgs.python3) (oldAttrs.buildInputs lib.or []))) ++ [ pythonEnv ];
  });
  juliaEnv = juliaEnvModBuildIn.withPackages.override {
    # For PyCAll to use our PythonEnv not some installed one
    makeWrapperArgs = [
      "--set" "PYTHONHOME" "${pythonEnv}"                    # Set the Python home to your desired Python environment
      "--prefix" "PYTHONPATH" ":" "${pythonEnv}/${pythonEnv.sitePackages}" # Include the site-packages in the Python path
      "--set" "PYTHON" "${pythonEnv}/bin/python3"            # Set the Python executable
    ];
    packageOverrides = {
      # Override SymbolicRegression with a specific version
      "SymbolicRegression" = pkgs.fetchFromGitHub {
        owner = "MilesCranmer";
        repo = "SymbolicRegression.jl";
        rev = "v1.5.2";
        sha256 = "sha256-DfNxd+N0dDk2Ex4/yEi/22L85LB+VJILOtMq1i/fdA8="; # Replace with actual hash
      };
      ## SymbolicRegression Dependency overwrites
      "DynamicQuantities" = pkgs.fetchFromGitHub {
        owner = "SymbolicML";
        repo = "DynamicQuantities.jl";
        rev = "v1.0.0";
        sha256 = "sha256-7Ae+Hv+S/c8NP1Wxj3h8c0r9nULTqoOO32LcOX6Oxrw="; # Replace with actual hash
      };
      "Compat" = pkgs.fetchFromGitHub {
        owner = "JuliaLang";
        repo = "Compat.jl";
        rev = "v4.16.0";
        sha256 = "sha256-bBAedBfVnqsMnJVKST9d9PTmmSuO7OBvptyhQ3YeGtg="; # Replace with actual hash
      };
      "DynamicDiff" = pkgs.fetchFromGitHub {
        owner = "MilesCranmer";
        repo = "DynamicDiff.jl";
        rev = "v0.2.0";
        sha256 = "sha256-V8FMZ2kcyirg0xm7hvCSdhbpenKMGkkdgUABqVseOpY="; # Replace with actual hash
      };
      "DispatchDoctor" = pkgs.fetchFromGitHub {
        owner = "MilesCranmer";
        repo = "DispatchDoctor.jl";
        rev = "v0.4.17";
        sha256 = "sha256-1x9b4hQFPMeMwXjh7J/Yy1M6JbGs9hNx1VJtqync2yY="; # Replace with actual hash
      };
      "DynamicExpressions" = pkgs.fetchFromGitHub {
        owner = "SymbolicML";
        repo = "DynamicExpressions.jl";
        rev = "v1.9.2";
        sha256 = "sha256-kNFBGcT3eEYC0KCautJ9zvmli/kPj2zOcR+NlvusoBI="; # Replace with actual hash
      };
    };
  } [
    # Remaining packages
    "DynamicExpressions" 
    "DispatchDoctor"
    "DynamicDiff"
    "Compat"
    "DynamicQuantities"
    "SymbolicRegression"
    "IJulia"
    "PyCall"
    "HDF5"
    "ProgressMeter"
    "PyPlot"
    "Latexify"
    "ClusterManagers" # not necessarily required but just to be sure
    "MLJBase"
    "LoopVectorization"
  ];
in
  

  pkgs.mkShell {
    name = "julia-python-env";

    buildInputs = [
      pkgs.hdf5 
      #pkgs.jupyterlab 
      pythonEnv
      juliaEnv
    ];
    # Because the jupyter kernel likes using the wrong paths we generate all of htis explicitly
    # Note that this deletes any other kernel with the same naming convention ie julia-version
    shellHook = ''
    # Dynamically resolve paths from the Nix store
    JULIA_BIN=$(readlink -f $(which julia))
    DEPOT_PATH=$(julia -e 'println(Sys.DEPOT_PATH[2])')
    PROJECT_PATH=$(julia -e 'println(Base.active_project())')
    JULIA_VERSION=$(julia -e 'println(VERSION)')
    KERNEL_DIR="$HOME/.local/share/jupyter/kernels/julia-$(julia -e 'println(VERSION)')"
    PYTHON=$(which python)
    export PYTHON_WITH_PACKAGES=$(which python)

    echo "Resolved paths:"
    echo "Built Environment: ${pythonEnv}"
    JULIA_PYCALL=$(julia -e 'import PyCall; println(PyCall.python)')
    echo "JuliaPyCall Env=$JULIA_PYCALL"

    # Remove Julia kernels with the same version - Avoid Spam
    for kernel in $HOME/.local/share/jupyter/kernels/julia-$JULIA_VERSION*; do
      if [ "$kernel" != "$KERNEL_DIR" ]; then
        echo "Removing conflicting kernel: $kernel"
        rm -rf "$kernel"
      fi
    done

    # Create kernel directory
    mkdir -p "$KERNEL_DIR"

    # Write kernel.json because it doesnt understand automatically - I wonder if this makes it work without the shell
    cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "$JULIA_BIN",
    "-i",
    "--color=yes",
    "--project=$PROJECT_PATH",
    "$DEPOT_PATH/packages/IJulia/bHdNn/src/kernel.jl",
    "{connection_file}"
  ],
  "display_name": "Julia $(julia -e 'println(VERSION)')",
  "language": "julia",
  "env": {
    "JULIA_DEPOT_PATH": "$DEPOT_PATH",
    "JULIA_LOAD_PATH": "$PROJECT_PATH:@stdlib"
  }
}
EOF

    echo "Kernel spec for Julia $(julia -e 'println(VERSION)') generated at $KERNEL_DIR/kernel.json"
  '';
}
