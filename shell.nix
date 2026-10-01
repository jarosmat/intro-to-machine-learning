with import <nixpkgs> {};
mkShell {
  packages = [
    python3
    python3Packages.matplotlib
    python3Packages.numpy
    python3Packages.scikit-learn
  ];
  shellHook = ''
    echo "Python interpreter path: $(which python)"
  '';
}
