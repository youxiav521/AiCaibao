from __future__ import annotations

from src.agents.policy.policy_check_agent.schemas import ResourceRepository

from .classifier import RuleBasedDocumentClassifier
from .extractor import CanonicalFieldExtractor
from .indexer import EvidenceIndexer
from .normalizer import MineruDocumentNormalizer
from .planner import ControlRetrievalPlanner
from .retriever import MaterialEvidenceRetriever
from .schemas import MaterialUnderstandingResult


class MaterialUnderstandingService:
    def __init__(self) -> None:
        self.classifier = RuleBasedDocumentClassifier()
        self.normalizer = MineruDocumentNormalizer()
        self.field_extractor = CanonicalFieldExtractor()
        self.indexer = EvidenceIndexer()
        self.planner = ControlRetrievalPlanner()

    def build(self, resource_repository: ResourceRepository) -> MaterialUnderstandingResult:
        documents = []
        for resource_file in resource_repository.files:
            classification = self.classifier.classify(resource_file)
            document = self.normalizer.normalize(resource_file, classification)
            document.canonicalFields = self.field_extractor.extract(document)
            documents.append(document)

        return MaterialUnderstandingResult(
            repositoryId=resource_repository.id,
            documents=documents,
            evidenceIndex=self.indexer.build(documents),
        )

    def build_retriever(self, resource_repository: ResourceRepository) -> MaterialEvidenceRetriever:
        return MaterialEvidenceRetriever(self.build(resource_repository))
