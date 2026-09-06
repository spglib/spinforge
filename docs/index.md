---
hide:
  - toc
---

# Forge magnetic structures from symmetry

SpinForge generates spin-symmetry-adapted and oriented magnetic crystal
structures from crystallographic symmetry. Use this page to choose a route
based on what you already know and what you need to do.

[Choose a route](#choose-your-route){ .md-button .md-button--primary }
[Install SpinForge](installation.md){ .md-button }

!!! note "Project scope"

    SpinForge generates symmetry-compatible candidates. It does not determine
    or refine a magnetic structure from experimental or first-principles data.

## Choose your route

You do not need to read the documentation from beginning to end. Each route
below states its prerequisite, reading order, and destination. Follow one route
until it meets your goal, then stop or branch to another.

<div class="grid cards route-grid" markdown>

-   :material-compass-outline:{ .lg .middle } **Understand the model**

    ---

    **Start here if:** SpinForge's domain vocabulary or representation of a
    magnetic structure is new to you.

    **Read:** [Domain model](domain-model.md) →
    [Classification model](classification-model.md)

    **You will be able to:** explain what SpinForge represents, which stages it
    distinguishes, and why different equivalence relations produce different
    candidate sets.

    **You can skip:** installation, tutorials, and API details if you only need
    the conceptual model.

-   :material-school-outline:{ .lg .middle } **Learn SpinForge**

    ---

    **Start here if:** you know the magnetic-crystallography concepts and are
    new to the package.

    **Read:** [Installation](installation.md) →
    [Your first structure](quickstart.md) → [Examples](examples.md)

    **You will be able to:** install SpinForge, enumerate a first set of
    candidates, and inspect the result.

    **You can skip:** the conceptual route when its terminology is already
    familiar. Return to the classification model if a result count or grouping
    is unexpected.

-   :material-tools:{ .lg .middle } **Complete a task**

    ---

    **Start here if:** SpinForge is installed and you can already run a basic
    enumeration.

    **Go directly to:** [Control enumeration](control-enumeration.md),
    [Use propagation vectors](propagation-vectors.md), or
    [Export files](export-files.md).

    **You will be able to:** change the search and equivalence settings or
    write the selected candidates for downstream use.

    **You can skip:** the tutorial and unrelated task guides.

-   :material-code-braces:{ .lg .middle } **Check exact behavior**

    ---

    **Start here if:** you are implementing against SpinForge or verifying an
    exact contract.

    **Go directly to:** [Enumeration and equivalence](equivalence.md),
    [File formats](file-formats.md), or the relevant
    [API reference](api/configuration.md). Consult the
    [Classification model](classification-model.md) when the meaning of a
    returned object or equivalence relation matters.

    **You will be able to:** confirm signatures, return types, object
    relationships, and classification semantics.

    **You can skip:** tutorials and task guides when you already know which API
    surface you need.

</div>
