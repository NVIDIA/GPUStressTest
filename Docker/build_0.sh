./populate_software.sh
cp ../build/gst software/
docker build -f Dockerfile_0 -t us-west2-docker.pkg.dev/egx-anthos-gpu-operator-val/gst/v2:ubuntu-18.04 .
