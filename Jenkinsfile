#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
// Keep only the most recent XX builds
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
  disableConcurrentBuilds()])

node ('rocmtest')
{
  def workspace_dir = pwd()
  def build_dir_debug = "${workspace_dir}/build/debug"
  def build_dir_release = "${workspace_dir}/build/release"

  withDockerContainer(image: 'hcc-lc-build-ubuntu-16.04', args: '--device=/dev/kfd')
  {
    stage('HCC Checkout')
    {
        checkout scm
        // git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git

        // The following common way to check out submodules is commented out because of the time it takes to update
        // the full history of the submodules
        sh 'git submodule update --init'
        // sh '''
        // git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc-clang-upgrade.git hcc/clang
        // git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/llvm.git hcc/compiler
        // git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/lld.git hcc/lld
        // git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/clang-tools-extra.git hcc/clang/tools/extra
        // '''
    }

    stage('hcc release build')
    {
      // Build release hcc
      dir("${build_dir_release}")
      {
        def build_config = "Release"
        def hcc_install_prefix = "/opt/rocm"
        sh  """#!/usr/bin/env bash
            sudo apt install file
            # build HCC with ROCm-Device-Libs
            cd ${build_dir_release}
            cmake \
              -DCMAKE_INSTALL_PREFIX=${hcc_install_prefix} \
              -DCMAKE_BUILD_TYPE=${build_config} \
              -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx803" \
              ${workspace_dir}
            make -j\$(nproc)
          """
      }

      stage("hcc package")
      {
        sh "cd ${build_dir_release}; make -j\$(nproc) package"
        archive includes: "${build_dir_release}/*.deb"
      }

      // Cap the maximum amount of testing, in case of hangs
      timeout(time: 1, unit: 'HOURS')
      {
        stage("hcc tests")
        {
          // install from debian packages because pre/post scripts set up softlinks install targets don't
          sh  """#!/usr/bin/env bash
              cd ${build_dir_release}
              dpkg -i hcc-*.deb
              echo Do reasonable build sanity tests here
              """
          // junit 'clients-build/tests-build/staging/*.xml'
        }
      }
    }
  }
}
