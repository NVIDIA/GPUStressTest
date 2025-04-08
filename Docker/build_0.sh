./populate_software.sh
cp ../build/gst software/
docker build -f Dockerfile_0 -t us-docker.pkg.dev/k80-exploration/gst/gst:v2.5.1-ubuntu-22.04 .

