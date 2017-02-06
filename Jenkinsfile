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
  // Convenience variables for common paths used in building
  def workspace_dir_abs = pwd()
  def build_dir_debug_rel = "build/debug"
  def build_dir_release_rel = "build/release"
  def build_dir_debug_abs = "${workspace_dir_abs}/${build_dir_debug_rel}"
  def build_dir_release_abs = "${workspace_dir_abs}/${build_dir_release_rel}"

  // The client workspace is shared with the docker container
  stage('HCC Checkout')
  {
    // git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc.git
    checkout scm
    sh 'git submodule update --init'

    // This is to build the extra clang tools
    sh '''
    rm -rf hcc/clang/tools/extra
    git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/clang-tools-extra.git hcc/clang/tools/extra
    '''
    // git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc-clang-upgrade.git hcc/clang
    // git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/llvm.git hcc/compiler
    // git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/lld.git hcc/lld
  }

  def hcc_build_image = null
  stage('ubuntu-16.04 image')
  {
    dir('docker')
    {
      hcc_build_image = docker.build( 'hcc-lc/build-ubuntu-16.04:latest', '-f dockerfile-build-ubuntu-16.04 --build-arg build_type=Release --build-arg rocm_install_path=/opt/rocm .' )
    }
  }

  hcc_build_image.inside( '--device=/dev/kfd' )
  {
    stage('hcc-lc release')
    {
      // Build release hcc
      dir("${build_dir_release_abs}")
      {
        deleteDir()
        def build_config = "Release"
        def hcc_install_prefix = "/opt/rocm"
        sh  """#!/usr/bin/env bash
            # build HCC with ROCm-Device-Libs
            cd ${build_dir_release_abs}
            cmake \
              -DCMAKE_INSTALL_PREFIX=${hcc_install_prefix} \
              -DCMAKE_BUILD_TYPE=${build_config} \
              -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx803" \
              ${workspace_dir_abs}
            make -j\$(nproc)
          """
      }

      // Cap the maximum amount of testing, in case of hangs
      timeout(time: 1, unit: 'HOURS')
      {
        stage("unit testing")
        {
          // install from debian packages because pre/post scripts set up softlinks install targets don't
          sh  """#!/usr/bin/env bash
              cd ${build_dir_release_abs}
              echo Do reasonable build sanity tests here
              """
          // junit "${build_dir_release_abs}/*.xml"
        }
      }

      stage("packaging")
      {
        sh "cd ${build_dir_release_abs}; make package"
        archiveArtifacts artifacts: "${build_dir_release_rel}/*.deb", fingerprint: true
      }
    }
  }

  // Everything above builds hcc in a clean container to create a debain package
  // Create a clean docker image that installs the debian package that was built.
  def hcc_install_image = null
  stage('hcc-lc image')
  {
    dir("${build_dir_release_abs}/docker")
    {
      sh "cp -r ${workspace_dir_abs}/docker/* .; cp ${build_dir_release_abs}/*.deb .; ls -la"
      // hcc_build_image = docker.build( "rocm/hcc-lc-ubuntu-16.04:${env.BUILD_TAG}", '-f dockerfile-hcc-lc-ubuntu-16.04 .' )
      hcc_build_image = docker.build( "rocm/hcc-lc-ubuntu-16.04:latest", '-f dockerfile-hcc-lc-ubuntu-16.04 .' )
    }
  }

}
