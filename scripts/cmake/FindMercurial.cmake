# - Extract information from a Mercurial repository
# The module defines the following variables:
#  MERCURIAL_EXECUTABLE - path to mercurial executable
#  MERCURIAL_VERSION - mercurial version
#  MERCURIAL_FOUND - true if mercurial was found
# If the mercurial executable is found the macro
#  MERCURIAL_HG_INFO(<dir> <var-prefix>)
# is defined to extract information of a mercurial repository at
# a given location. The macro defines the following variables:
#  <var-prefix>_HG_ID - global revision id of the working copy (at <dir>)
#  <var-prefix>_HG_CHANGESET - changeset id of the working copy
#  <var-prefix>_HG_AUTHOR - commit author of this changeset
#  <var-prefix>_HG_DATE - commit date of this changeset
#  <var-prefix>_HG_TAGS - tags belonging to this changeset
#  <var-prefix>_HG_BRANCH - (non-default) branch name of this changeset
#  <var-prefix>_HG_SUMMARY - commit message summary of this changeset
# Example usage:
#  FIND_PACKAGE(Mercurial)
#  IF(MERCURIAL_FOUND)
#    MERCURIAL_HG_INFO(${PROJECT_SOURCE_DIR} Project)
#    MESSAGE("Current revision is ${Project_HG_ID}")
#  ENDIF(MERCURIAL_FOUND)

# Copyright (C) 2008  Peter Colberg
#
# This file was derived from FindSubversion.cmake shipped with CMake 2.4.7.
#
# Copyright (c) 2006, Tristan Carel
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of California, Berkeley nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


FIND_PROGRAM(MERCURIAL_EXECUTABLE hg
  DOC "Mercurial Distributed SCM executable")
MARK_AS_ADVANCED(MERCURIAL_EXECUTABLE)

IF(MERCURIAL_EXECUTABLE)
  SET(MERCURIAL_FOUND TRUE)

  MACRO(MERCURIAL_COMMAND dir command)
    EXECUTE_PROCESS(COMMAND ${MERCURIAL_EXECUTABLE} ${command} ${ARGN}
      WORKING_DIRECTORY ${dir}
      OUTPUT_VARIABLE MERCURIAL_${command}_OUTPUT
      ERROR_VARIABLE MERCURIAL_${command}_ERROR
      RESULT_VARIABLE MERCURIAL_${command}_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    IF(NOT ${MERCURIAL_${command}_RESULT} EQUAL 0)
      SET(cmdline "${MERCURIAL_EXECUTABLE} ${command}")
      FOREACH(arg ${ARGN})
  SET(cmdline "${cmdline} ${arg}")
      ENDFOREACH(arg ${ARGN})
      MESSAGE(SEND_ERROR "Command \"${cmdline}\" failed with output:\n${MERCURIAL_${command}_ERROR}")

      SET(MERCURIAL_${command}_OUTPUT)
    ENDIF(NOT ${MERCURIAL_${command}_RESULT} EQUAL 0)

  ENDMACRO(MERCURIAL_COMMAND dir command)


  MACRO(MERCURIAL_HG_INFO dir prefix)
    IF(IS_DIRECTORY "${dir}")
      MERCURIAL_COMMAND(${dir} id -i)

      IF(MERCURIAL_id_OUTPUT)
  # global revision id of the working copy
  SET(${prefix}_HG_ID "${MERCURIAL_id_OUTPUT}")

  # changeset id of the working copy
  STRING(REGEX REPLACE "^([0-9a-f]+).*"
    "\\1" ${prefix}_HG_CHANGESET "${MERCURIAL_id_OUTPUT}")

  MERCURIAL_COMMAND(${dir} log -r ${${prefix}_HG_CHANGESET})

  STRING(REGEX REPLACE ";" "\\\\;"
    MERCURIAL_log_OUTPUT "${MERCURIAL_log_OUTPUT}")
  STRING(REGEX REPLACE "\n" ";"
    MERCURIAL_log_OUTPUT "${MERCURIAL_log_OUTPUT}")

  FOREACH(line ${MERCURIAL_log_OUTPUT})
    # commit author of this changeset
    IF(line MATCHES "^user:")
      STRING(REGEX REPLACE "^user:[ ]+(.+)"
        "\\1" ${prefix}_HG_AUTHOR "${line}")
    ENDIF(line MATCHES "^user:")

    # commit date of this changeset
    IF(line MATCHES "^date:")
      STRING(REGEX REPLACE "^date:[ ]+(.+)"
        "\\1" ${prefix}_HG_DATE "${line}")
    ENDIF(line MATCHES "^date:")

    # tags belonging to this changeset
    IF(line MATCHES "^tag:")
      STRING(REGEX REPLACE "^tag:[ ]+(.+)"
        "\\1" tag "${line}")
      STRING(REGEX REPLACE ";" "\\\\;" tag "${tag}")
      LIST(APPEND ${prefix}_HG_TAGS "${tag}")
    ENDIF(line MATCHES "^tag:")

    # (non-default) branch name of this changeset
    IF(line MATCHES "^branch:")
      STRING(REGEX REPLACE "^branch:[ ]+(.+)"
        "\\1" ${prefix}_HG_BRANCH "${line}")
    ENDIF(line MATCHES "^branch:")

    # commit message summary of this changeset
    IF(line MATCHES "^summary:")
      STRING(REGEX REPLACE "^summary:[ ]+(.+)"
        "\\1" ${prefix}_HG_SUMMARY "${line}")
    ENDIF(line MATCHES "^summary:")

  ENDFOREACH(line ${MERCURIAL_log_OUTPUT})
      ENDIF(MERCURIAL_id_OUTPUT)

    ELSE(IS_DIRECTORY "${dir}")
      MESSAGE(SEND_ERROR "Invalid MERCURIAL_HG_INFO directory \"${dir}\"")
    ENDIF(IS_DIRECTORY "${dir}")

  ENDMACRO(MERCURIAL_HG_INFO dir prefix)


  # mercurial version
  MERCURIAL_COMMAND(${CMAKE_BINARY_DIR} version)

  STRING(REGEX REPLACE "^Mercurial Distributed SCM \\(version ([.0-9]+)\\).*"
    "\\1" MERCURIAL_VERSION "${MERCURIAL_version_OUTPUT}")

ENDIF(MERCURIAL_EXECUTABLE)

IF(NOT MERCURIAL_FOUND)
  IF(NOT MERCURIAL_FIND_QUIETLY)
    MESSAGE(STATUS "Mercurial was not found.")
  ELSE(NOT MERCURIAL_FIND_QUIETLY)
    IF(MERCURIAL_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Mercurial was not found.")
    ENDIF(MERCURIAL_FIND_REQUIRED)
  ENDIF(NOT MERCURIAL_FIND_QUIETLY)
ENDIF(NOT MERCURIAL_FOUND)
