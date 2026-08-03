# Kozax Assimilation Overview

This documentation details the process of assimilating the [Kozax](https://github.com/sdevries0/Kozax) framework into MalthusJAX.

## Framework Identity
- **Name**: Kozax
- **Target Domain**: Genetic Programming (Symbolic Regression) in JAX.
- **Core Abstraction**: An monolithic `evolve_population` loop that modifies a population tensor of expression trees directly.

## Goal

The objective is to utilize MalthusJAX's Universal `@adapter` to wrap Kozax's Genetic Programming solver. This enables MalthusJAX to leverage Kozax's sophisticated JAX-based expression tree encodings while routing fitness evaluations either natively to Kozax's mesh evaluation, or into MalthusJAX for modular environment simulation.
