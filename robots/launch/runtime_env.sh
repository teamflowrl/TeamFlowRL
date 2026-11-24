# Optional runtime environment for headless ManiSkill / RoboFactory rendering on GPU nodes.
# Source it before training, or pass it to the launchers as RUNTIME_ENV=robots/launch/runtime_env.sh.
#   MANISKILL_SRC  ManiSkill source checkout to put on PYTHONPATH (optional)
#   RUNTIME_LIBS   directory with the Vulkan/EGL runtime libraries and ICD json files (optional)
if [ -n "${MANISKILL_SRC:-}" ]; then export PYTHONPATH=${MANISKILL_SRC}${PYTHONPATH:+:$PYTHONPATH}; fi
if [ -n "${RUNTIME_LIBS:-}" ]; then
  export LD_LIBRARY_PATH=${RUNTIME_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export VK_ICD_FILENAMES=${RUNTIME_LIBS}/nvidia_icd.json
  export __EGL_VENDOR_LIBRARY_FILENAMES=${RUNTIME_LIBS}/10_nvidia.json
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONFAULTHANDLER=1
# Optional: preload a driver-specific library if the Vulkan runtime needs it.
if [ -n "${GPU_PRELOAD_LIB:-}" ]; then export LD_PRELOAD=${GPU_PRELOAD_LIB}; fi
