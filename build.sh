#!/bin/bash

workdir=`pwd`

if [ -z "$PPL_BUILD_THREAD_NUM" ]; then
    PPL_BUILD_THREAD_NUM=1
    echo -e "env 'PPL_BUILD_THREAD_NUM' is not set. use PPL_BUILD_THREAD_NUM=${PPL_BUILD_THREAD_NUM} by default."
fi

build_type='Release'
options="$*"

ppl_build_dir="${workdir}/ppl-build"
mkdir ${ppl_build_dir}
cd ${ppl_build_dir}
cmd="cmake $options .. && cmake --build . -j ${PPL_BUILD_THREAD_NUM} --config ${build_type}"
# cmd="cmake $options .. && cmake --build . --verbose -j ${PPL_BUILD_THREAD_NUM} --config ${build_type}"
echo "cmd -> $cmd"
eval "$cmd"
