# TensorNEAT Assimilation Overview

This documentation details the process of assimilating the [TensorNEAT](https://github.com/EMI-Group/tensorneat) framework into MalthusJAX.

## Framework Identity
- **Name**: TensorNEAT
- **Target Domain**: NeuroEvolution of Augmenting Topologies (NEAT, CPPN, HyperNEAT) entirely on GPU.
- **Core Abstraction**: An "Ask-Transform-Eval-Tell" functional loop utilizing customized `tensorneat.common.State` dictionary wrappers.

## Goal

The objective is to utilize MalthusJAX's Universal `@adapter` to wrap TensorNEAT algorithms (like `tensorneat.algorithm.NEAT`) so that they can be transparently executed by the MalthusJAX engine. This provides the ability to evolve neural network topologies and weights using MalthusJAX Evaluators without being bound to TensorNEAT's native pipeline.
