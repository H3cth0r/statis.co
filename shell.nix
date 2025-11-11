
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "statisco-env";

  buildInputs = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv

    # native libraries for numpy, pandas, etc.
    pkgs.zlib
    pkgs.libffi
    pkgs.glib
    pkgs.openssl
    pkgs.stdenv.cc.cc        # includes libstdc++.so.6 and libgcc_s.so.1
  ];

  shellHook = ''
    echo "🔧 Setting up Python environment for statis.co..."

    # ensure lib paths are visible to dynamic linker
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glib}/lib:${pkgs.openssl.out}/lib

    if [ ! -d .venv ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip setuptools wheel
      if [ -f requirements.txt ]; then
        echo "📦 Installing from requirements.txt..."
        pip install -r requirements.txt
      fi
      if [ -f pyproject.toml ]; then
        echo "📦 Installing editable project..."
        pip install -e .
      fi
    else
      source .venv/bin/activate
    fi

    echo "✅ Environment ready. Use 'deactivate' to exit."
  '';
}
