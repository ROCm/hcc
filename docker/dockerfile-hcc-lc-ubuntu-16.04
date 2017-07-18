FROM ubuntu:16.04
MAINTAINER Kent Knox <kent.knox@amd>

# Parameters related to building hcc-lc
ARG REPO_RADEON=repo.radeon.com

# Copy the debian package of hcc-lc into the container from host
COPY *.deb /tmp/

# NOTE: Make sure apt-get update and apt-get install are all on the same RUN command
# Seperate RUN commands are cached by docker in different layers, and the contents of the update command need to
# be refreshed every time the container is built, or repo metadata will get stale and packages it is looking for
# won't exist upstream
# FIXME: add --allow-unauthenticated to workaround issue with $REPO_RADEON
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y curl \
  && curl -sL http://${REPO_RADEON}/rocm/apt/debian/rocm.gpg.key | apt-key add - \
  && echo deb [arch=amd64] http://${REPO_RADEON}/rocm/apt/debian/ xenial main | tee /etc/apt/sources.list.d/rocm.list \
  && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --allow-unauthenticated -y \
    /tmp/hcc*.deb \
  && rm -f /tmp/*.deb \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
