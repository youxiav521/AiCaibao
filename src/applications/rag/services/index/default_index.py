from src.applications.rag.services.summary_index import build_summary_index
from src.applications.rag.chunking.grandchildren import create_grandchildren
from src.applications.rag.schemas import ChildChunk, GrandchildChunk, ParentChunk
from .index_writer import write_index_payloads
from src.utils.stopwatch import StopWatch


class DefaultIndex:
    def __init__(
        self,
        file_id: str,
        mineru_json: list[dict],
        folder_id: str | None = None,
        file_name: str | None = None,
    ):
        self.file_id = file_id
        self.mineru_json = mineru_json
        self.folder_id = folder_id
        self.file_name = file_name
        self.stopwatch = StopWatch(f"Default-Index-{file_id}")

    async def summary_index(self):
        with self.stopwatch.task("summary_index") as t:
            parents, children, summary_grandchildren = await build_summary_index(
                self.mineru_json,
                file_id=self.file_id,
                folder_id=self.folder_id,
                file_name=self.file_name,
                build_summaries=True,
                emit_summary_grandchildren=False,
            )

        grandchildren = create_grandchildren(
            children,
            self.mineru_json,
            file_id=self.file_id,
            folder_id=self.folder_id,
            file_name=self.file_name,
        )
        t.detail(
            f"{len(parents)} parents, {len(children)} children, "
            + f"{len(grandchildren)} grandchildren, "
            + f"{len(summary_grandchildren)} summary grandchildren"
        )
        self.parents = parents
        self.children = children
        self.grandchildren = grandchildren
        self.summary_grandchildren = summary_grandchildren

    async def save_summary_index(self):
        with self.stopwatch.task("write_summary_index") as t:
            total = (
                len(self.parents)
                + len(self.children)
                + len(self.grandchildren)
                + len(self.summary_grandchildren)
            )
            t.detail(f"{total} documents")
            return await write_index_payloads(
                file_id=self.file_id,
                parents=self.parents,
                children=self.children,
                grandchildren=self.grandchildren + self.summary_grandchildren,
            )

    async def build(self):
        await self.summary_index()
        res = await self.save_summary_index()
        self.stopwatch.print()

        parent_documents: list[ParentChunk] = []
        child_documents: list[ChildChunk] = []
        grandchild_documents: list[GrandchildChunk] = []

        if isinstance(res, tuple) and len(res) == 3:
            p, c, g = res
            parent_documents.extend(p or [])
            child_documents.extend(c or [])
            grandchild_documents.extend(g or [])

        return parent_documents, child_documents, grandchild_documents


async def build(
    file_id: str,
    mineru_json: list[dict],
    folder_id: str | None = None,
    file_name: str | None = None,
):
    return await DefaultIndex(
        file_id=file_id,
        mineru_json=mineru_json,
        folder_id=folder_id,
        file_name=file_name,
    ).build()
