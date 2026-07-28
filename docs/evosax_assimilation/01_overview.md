# Evosax Assimilation Overview

This documentation details the process of assimilating the [evosax](https://github.com/RobertTLange/evosax) framework into MalthusJAX.

## Framework Identity
- **Name**: Evosax
- **Target Domain**: Evolution Strategies (ES) and Black-Box Optimization (BBO).
- **Core Abstraction**: An "Ask-Eval-Tell" functional loop operating on standard JAX PRNGKeys and pure PyTree states.

## Goal

The objective is to utilize MalthusJAX's Universal `@adapter` to wrap Evosax's `Strategy` objects (e.g., `CMA_ES`, `OpenES`, `DE`) such that they can be transparently executed by the MalthusJAX engine. This allows users to leverage Evosax's high-performance search algorithms while retaining the option to evaluate candidates using MalthusJAX's native fitness evaluators or Evosax's native problems (like BBOB).
