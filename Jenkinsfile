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
    deleteDir( )
    checkout scm

    // submodule update may not be updating fresh files from submodules
    // sh 'git submodule update --init'

    // Manually clone all submodules to get shallow copies to speed up checkout time
    sh  '''
        git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc-clang-upgrade.git clang
        git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/llvm.git compiler
        git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/lld.git lld
        git clone --depth 1 -b amd-hcc https://github.com/RadeonOpenCompute/compiler-rt.git compiler-rt
        git clone --depth 1 -b clang_tot_upgrade https://github.com/RadeonOpenCompute/clang-tools-extra.git clang/tools/extra
        '''
  }

  def hcc_build_image = null
  stage('ubuntu-16.04 image')
  {
    def build_org = "hcc-lc"
    def build_image_name = "build-ubuntu-16.04"
    dir('docker')
    {
      hcc_build_image = docker.build( "${build_org}/${build_image_name}:latest", "-f dockerfile-${build_image_name} --build-arg build_type=Release --build-arg rocm_install_path=/opt/rocm ." )
    }
  }

// JENKINS-33510: the jenkinsfile dir() command is not workin well with docker.inside()
  hcc_build_image.inside( '--device=/dev/kfd' )
  {
    stage('hcc-lc release')
    {
      // Build release hcc
      def build_config = "Release"
      def hcc_install_prefix = "/opt/rocm"

      // cmake -B${build_dir_release_abs} specifies to cmake where to generate build files
      // This is necessary because cmake seemingly randomly generates build makefile into the docker
      // workspace instead of the current set directory.  Not sure why, but it seems like a bug
      sh  """
          mkdir -p ${build_dir_release_rel}
          cd ${build_dir_release_rel}
          cmake -B${build_dir_release_abs} \
            -DCMAKE_INSTALL_PREFIX=${hcc_install_prefix} \
            -DCMAKE_BUILD_TYPE=${build_config} \
            -DHSA_AMDGPU_GPU_TARGET="gfx701;gfx803" \
            ../..
          make -j\$(nproc)
        """

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
  // Create a clean docker image that installs the debian package
  def hcc_install_image = null
  stage('artifactory')
  {
    def artifactory_org = "${env.JOB_NAME}".toLowerCase()
    def image_name = "hcc-lc-ubuntu-16.04"

    dir("${build_dir_release_abs}/docker")
    {
      //  We copy the docker files into the bin directory where the .deb lives so that it's a clean
      //  build everytime
      sh "cp -r ${workspace_dir_abs}/docker/* .; cp ${build_dir_release_abs}/*.deb ."
      hcc_build_image = docker.build( "${artifactory_org}/${image_name}:${env.BUILD_NUMBER}", "-f dockerfile-${image_name} ." )
    }

    docker.withRegistry('http://compute-artifactory:5001', 'artifactory-cred' )
    {
      hcc_build_image.push( "${env.BUILD_NUMBER}" )
      hcc_build_image.push( 'latest' )
    }

    // Lots of images with tags are created above; no apparent way to delete images:tags with docker global variable
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${artifactory_org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }

}
