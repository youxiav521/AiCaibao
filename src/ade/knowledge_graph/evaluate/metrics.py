from pathlib import Path

from src.ade.knowledge_graph.evaluate.parser import parse_graph
from src.ade.knowledge_graph.evaluate.schema import (
    ENTERPRISE_ATTR_ALIASES,
    EXPECTED_TO_ENTITY,
    KEY_RELATION_TYPES,
    AttributeAtom,
    AttributeEvalResult,
    AttributeKey,
    ConfidenceDetectionResult,
    ConfidenceDetectionSummary,
    ConfidenceEvaluationResult,
    ConfidenceLevel,
    EntityKey,
    EvaluationEntity,
    EvaluationGraph,
    EvaluationOptions,
    EvaluationRelation,
    EvaluationResult,
    GraphFormat,
    MetricResult,
    RelationAttributeAtom,
    RelationAttributeEvalResult,
    RelationAttributeKey,
    RelationTriple,
)


def run_evaluation(
    gold_path: Path,
    target_path: Path,
    *,
    gold_format: GraphFormat = "auto",
    target_format: GraphFormat = "auto",
    options: EvaluationOptions | None = None,
) -> EvaluationResult:
    opts = options or EvaluationOptions()
    gold = parse_graph(gold_path, gold_format)
    target = parse_graph(target_path, target_format)
    key_total = evaluate_key_relations(gold, target)
    return EvaluationResult(
        gold_path=gold_path,
        target_path=target_path,
        gold=gold,
        target=target,
        entity_all=evaluate_entities(gold, target),
        relation=evaluate_relations(gold, target),
        attribute=evaluate_attributes(gold, target) if opts.include_attributes else None,
        relation_attribute=(
            evaluate_relation_attributes(gold, target)
            if opts.include_attributes
            else None
        ),
        key_relation_total=key_total,
        confidence_exclude_low=evaluate_confidence_filter(gold, target, {"Low"}),
        confidence_high_only=evaluate_confidence_filter(
            gold, target, {"Low", "Medium"}
        ),
        confidence_detection=evaluate_confidence_detection(gold, target),
        options=opts,
    )


def split_relation_types(type_text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in type_text.replace("，", ",").split(","):
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def entity_key(entity: EvaluationEntity) -> EntityKey:
    return (entity.name, entity.type_en)


def relation_triples(relations: list[EvaluationRelation]) -> set[RelationTriple]:
    triples: set[RelationTriple] = set()
    for relation in relations:
        for relation_type in split_relation_types(relation.type):
            triples.add((relation.from_entity, relation_type, relation.to_entity))
    return triples


def evaluate_sets(gold_set: set, target_set: set) -> MetricResult:
    correct = gold_set & target_set
    missing = gold_set - target_set
    extra = target_set - gold_set
    return MetricResult(
        tp=len(correct),
        fp=len(extra),
        fn=len(missing),
        correct=correct,
        missing=missing,
        extra=extra,
    )


def evaluate_entities(gold: EvaluationGraph, target: EvaluationGraph) -> MetricResult:
    gold_keys = {entity_key(e) for e in gold.entities}
    target_keys = {entity_key(e) for e in target.entities}
    return evaluate_sets(gold_keys, target_keys)


def evaluate_relations(gold: EvaluationGraph, target: EvaluationGraph) -> MetricResult:
    return evaluate_sets(relation_triples(gold.relations), relation_triples(target.relations))


def evaluate_confidence_filter(
    gold: EvaluationGraph,
    target: EvaluationGraph,
    excluded: set[ConfidenceLevel],
) -> ConfidenceEvaluationResult:
    removed_entity_names = {
        entity.name for entity in target.entities if entity.confidence in excluded
    }
    filtered = EvaluationGraph(
        entities=[
            entity for entity in target.entities if entity.confidence not in excluded
        ],
        relations=[
            relation
            for relation in target.relations
            if relation.confidence not in excluded
            and relation.from_entity not in removed_entity_names
            and relation.to_entity not in removed_entity_names
        ],
    )
    return ConfidenceEvaluationResult(
        entity=evaluate_entities(gold, filtered),
        relation=evaluate_relations(gold, filtered),
        key_relation=evaluate_key_relations(gold, filtered),
    )


def evaluate_confidence_detection(
    gold: EvaluationGraph, target: EvaluationGraph
) -> ConfidenceDetectionSummary:
    flagged_levels = {"Low", "Medium"}
    gold_entities = {entity_key(entity) for entity in gold.entities}
    target_entities = {entity_key(entity) for entity in target.entities}
    flagged_entities = {
        entity_key(entity)
        for entity in target.entities
        if entity.confidence in flagged_levels
    }
    gold_relations = relation_triples(gold.relations)
    target_relations = relation_triples(target.relations)
    flagged_relations = {
        triple
        for relation in target.relations
        if relation.confidence in flagged_levels
        for triple in relation_triples([relation])
    }
    entity_result = _confidence_detection_result(
        target_entities - gold_entities, flagged_entities
    )
    relation_result = _confidence_detection_result(
        target_relations - gold_relations, flagged_relations
    )
    return ConfidenceDetectionSummary(
        entity=entity_result,
        relation=relation_result,
        overall=ConfidenceDetectionResult(
            true_errors=entity_result.true_errors + relation_result.true_errors,
            flagged=entity_result.flagged + relation_result.flagged,
            flagged_errors=(
                entity_result.flagged_errors + relation_result.flagged_errors
            ),
        ),
    )


def _confidence_detection_result(
    true_errors: set, flagged: set
) -> ConfidenceDetectionResult:
    return ConfidenceDetectionResult(
        true_errors=len(true_errors),
        flagged=len(flagged),
        flagged_errors=len(true_errors & flagged),
    )


def evaluate_attributes(gold: EvaluationGraph, target: EvaluationGraph) -> AttributeEvalResult:
    matched_keys = {entity_key(e) for e in gold.entities} & {entity_key(e) for e in target.entities}
    gold_sets = _collect_attribute_value_sets(gold.entities, matched_keys)
    target_sets = _collect_attribute_value_sets(target.entities, matched_keys)

    correct: set[AttributeAtom] = set()
    missing: set[AttributeAtom] = set()
    extra: set[AttributeAtom] = set()
    value_wrong: list[tuple[AttributeKey, set[str], set[str]]] = []

    for key in set(gold_sets) | set(target_sets):
        gold_values = gold_sets.get(key, set())
        target_values = target_sets.get(key, set())
        correct_values = gold_values & target_values
        missing_values = gold_values - target_values
        extra_values = target_values - gold_values

        name, type_en, attr_key = key
        correct.update((name, type_en, attr_key, value) for value in correct_values)
        missing.update((name, type_en, attr_key, value) for value in missing_values)
        extra.update((name, type_en, attr_key, value) for value in extra_values)

        if gold_values and target_values and not correct_values:
            value_wrong.append((key, gold_values, target_values))

    return AttributeEvalResult(
        metrics=MetricResult(
            tp=len(correct),
            fp=len(extra),
            fn=len(missing),
            correct=correct,
            missing=missing,
            extra=extra,
        ),
        value_wrong=value_wrong,
    )


def evaluate_relation_attributes(
    gold: EvaluationGraph, target: EvaluationGraph
) -> RelationAttributeEvalResult:
    matched_triples = relation_triples(gold.relations) & relation_triples(target.relations)
    gold_sets = _collect_relation_attribute_value_sets(gold.relations, matched_triples)
    target_sets = _collect_relation_attribute_value_sets(target.relations, matched_triples)

    correct: set[RelationAttributeAtom] = set()
    missing: set[RelationAttributeAtom] = set()
    extra: set[RelationAttributeAtom] = set()
    value_wrong: list[tuple[RelationAttributeKey, set[str], set[str]]] = []

    for key in set(gold_sets) | set(target_sets):
        gold_values = gold_sets.get(key, set())
        target_values = target_sets.get(key, set())
        correct_values = gold_values & target_values
        missing_values = gold_values - target_values
        extra_values = target_values - gold_values

        from_entity, relation_type, to_entity, attr_key = key
        correct.update(
            (from_entity, relation_type, to_entity, attr_key, value)
            for value in correct_values
        )
        missing.update(
            (from_entity, relation_type, to_entity, attr_key, value)
            for value in missing_values
        )
        extra.update(
            (from_entity, relation_type, to_entity, attr_key, value)
            for value in extra_values
        )

        if gold_values and target_values and not correct_values:
            value_wrong.append((key, gold_values, target_values))

    return RelationAttributeEvalResult(
        metrics=MetricResult(
            tp=len(correct),
            fp=len(extra),
            fn=len(missing),
            correct=correct,
            missing=missing,
            extra=extra,
        ),
        value_wrong=value_wrong,
    )


def evaluate_key_relations(gold: EvaluationGraph, target: EvaluationGraph) -> MetricResult:
    gold_triples = relation_triples(gold.relations)
    target_triples = relation_triples(target.relations)
    return evaluate_sets(
        _normal_key_relations(gold_triples),
        _normal_key_relations(target_triples),
    )


def abnormal_key_relations(graph: EvaluationGraph) -> set[RelationTriple]:
    return _abnormal_key_relations(relation_triples(graph.relations))


def normalize_attr_key(entity_type_en: str, attr_key: str) -> str:
    if entity_type_en != "ENTERPRISE" or attr_key == "公章编码":
        return attr_key
    return ENTERPRISE_ATTR_ALIASES.get(attr_key, attr_key)


def _collect_attribute_value_sets(
    entities: list[EvaluationEntity], matched_keys: set[EntityKey]
) -> dict[AttributeKey, set[str]]:
    value_sets: dict[AttributeKey, set[str]] = {}
    for entity in entities:
        if entity_key(entity) not in matched_keys:
            continue
        for attr_key, attr_value in entity.attributes.items():
            normalized = normalize_attr_key(entity.type_en, attr_key)
            value_sets.setdefault((entity.name, entity.type_en, normalized), set()).add(attr_value)
    return value_sets


def _collect_relation_attribute_value_sets(
    relations: list[EvaluationRelation],
    matched_triples: set[RelationTriple],
) -> dict[RelationAttributeKey, set[str]]:
    value_sets: dict[RelationAttributeKey, set[str]] = {}
    for relation in relations:
        for relation_type in split_relation_types(relation.type):
            triple = (relation.from_entity, relation_type, relation.to_entity)
            if triple not in matched_triples:
                continue
            for attr_key, attr_values in relation.attributes.items():
                value_sets.setdefault((*triple, attr_key), set()).update(attr_values)
    return value_sets


def _normal_key_relations(triples: set[RelationTriple]) -> set[RelationTriple]:
    return {
        triple
        for triple in triples
        if triple[1] in KEY_RELATION_TYPES and triple[2] == EXPECTED_TO_ENTITY
    }


def _abnormal_key_relations(triples: set[RelationTriple]) -> set[RelationTriple]:
    return {
        triple
        for triple in triples
        if triple[1] in KEY_RELATION_TYPES and triple[2] != EXPECTED_TO_ENTITY
    }
