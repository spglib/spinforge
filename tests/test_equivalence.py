from spinforge.utils._equivalence import _classify_equivalence_classes


def test_classify_equivalence_classes_partitions_inputs() -> None:
    candidates = ["even:0", "even:2", "odd:1"]

    def find_transformation(candidate: str, representative: str) -> str | None:
        if candidate.partition(":")[0] != representative.partition(":")[0]:
            return None
        return f"{candidate} -> {representative}"

    equivalence_classes = _classify_equivalence_classes(
        candidates,
        find_transformation,
        lambda representative: [f"fix {representative}"],
    )

    assert [equivalence_class.representative for equivalence_class in equivalence_classes] == [
        "even:0",
        "odd:1",
    ]
    assert [equivalence_class.equivalent_objects for equivalence_class in equivalence_classes] == [
        ("even:0", "even:2"),
        ("odd:1",),
    ]
    assert [
        equivalence_class.transformations_to_representative
        for equivalence_class in equivalence_classes
    ] == [
        ("even:0 -> even:0", "even:2 -> even:0"),
        ("odd:1 -> odd:1",),
    ]
    assert [equivalence_class.stabilizer for equivalence_class in equivalence_classes] == [
        ("fix even:0",),
        ("fix odd:1",),
    ]
