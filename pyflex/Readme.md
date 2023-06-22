# Pyflex

This readme describes how to compile Nvidia Flex & the python bindings (PyFlex) and how to use them for simulating clothes.



# Acknowledgements

[PyFlex](https://github.com/YunzhuLi/PyFleX) and [Nvidia-Flex](https://developer.nvidia.com/flex-example) have a long history in robotics cloth manipulation. Though nividia-flex is discontinued (in favor of omniverse, which also has a cloth simulator) it remains attractive as it is open-source, cuda-accelerated (fast!) and stable.
[Softgym](https://sites.google.com/view/softgym) has leveraged this to build a suite of gym (now gymnasium)-like environments for RL agents. Their codebase was later used by the [FlingBot](https://flingbot.cs.columbia.edu/)paper, where they extended the cloth scene to create random crumpled states and learn to unfold them using dynamic fling primitive. [Cloth Funneling](https://clothfunnels.cs.columbia.edu/)  has adapted the flingbot Pyflex codebase to support their work on unfolding with multiple primitives.

  We build on top of the Cloth Funneling Pyflex codedbase and made modifications where required, such as [this one](https://github.com/columbia-ai-robotics/cloth-funnels/issues/3).



