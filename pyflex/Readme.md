# Pyflex

This readme describes how to compile Nvidia Flex & the python bindings (PyFlex) and how to use them for simulating clothes.

## Installation

> You need an NVIDIA GPU to run Nvidia-Flex!

To compile Nivida-flex and the PyFlex bindings, you need to use the provided Dockerfile. You will then be able to run Flex in your conda environment.

navigate to the `pyflex` folder.

Build the docker container
```docker build -t pyflex .```

Then run it (make sure you have the nvidia-docker toolkit installed so that your GPUS are exposed in docker):
```
docker run -v $PWD:/workspace/pyflex -v <path-to-your-conda-installation>:<path-to-your-conda-installation> --gpus all --shm-size=64gb  -d -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -it pyflex
```

For example, on my laptop this becomes:
```
docker run -v $PWD:/workspace/pyflex -v /home/tlips/miniconda3:/home/tlips/miniconda3 --gpus all --shm-size=64gb  -d -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -it pyflex
```
Check the container is running using `docker ps`.
Now you can attach to the container using

```
docker exec -it  <containe-name/ID> bash
```

Inside the container, activate your conda environment
```
source <conda-installation-path>/bin/activate
conda activate <your-env>
And then install the requirements for
Then set the required paths with
```
source ./prepare.sh
```

Finally, you can compile:
```
./compile.sh
```

Which will result in a dynamic library (.so file) in the `Pyflex/bindings/build` folder. This will then be loaded in python (if it is on the python path). See [pybind docs](https://pybind11.readthedocs.io/en/latest/basics.html) for more information.


You can now exit the docker container and use pyflex from your conda environment. Test it using:

```
source ./prepare.sh
python pyflex_utils/test_installation.py
```

This should print something like  (but with your GPU)
```
Compute Device: NVIDIA GeForce GTX 1050 Ti with Max-Q Design

Pyflex init done!
```

## Using Pyflex
The available API can be found at `bindings/pyflex.cpp`. You can extend this API or create additional scenes if you need. Don't forget to recompile using the above steps.

Note that the codebase has quite some code debt. AFAIK everything is built on top of the demo application, to which different scenes have been added over time. On top of that, a number of python bindings have been created in the pybind file. Basically everyone has hacked what he/she needed and then continued working, present company included. Doesn't look nice, but it gets the job done..

To understand the particle-based simulator and it's API, you can take a look at the  [mandual](https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/flex/manual.html#manual) and/or [paper](http://blog.mmacklin.com/project/flex/).

# Acknowledgements

[PyFlex](https://github.com/YunzhuLi/PyFleX) and [Nvidia-Flex](https://developer.nvidia.com/flex-example) have a long history in robotics cloth manipulation. Though nividia-flex is discontinued (in favor of omniverse, which also has a cloth simulator) it remains attractive as it is open-source, cuda-accelerated (fast!) and stable.
[Softgym](https://sites.google.com/view/softgym) has leveraged this to build a suite of gym (now gymnasium)-like environments for RL agents. Their codebase was later used by the [FlingBot](https://flingbot.cs.columbia.edu/)paper, where they extended the cloth scene to create random crumpled states and learn to unfold them using dynamic fling primitive. [Cloth Funneling](https://clothfunnels.cs.columbia.edu/)  has adapted the flingbot Pyflex codebase to support their work on unfolding with multiple primitives.

  We build on top of the Cloth Funneling Pyflex codedbase and made modifications where required, such as [this one](https://github.com/columbia-ai-robotics/cloth-funnels/issues/3).



