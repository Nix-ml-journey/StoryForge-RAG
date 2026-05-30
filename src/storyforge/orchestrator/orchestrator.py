import logging
from typing import Optional

from storyforge.config.config import CONFIG_FILE, load_config

from storyforge.rag.generative_ai import Gen_mode, StoryType
from . import response_parameter as parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Orchestrator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        c = self.config
        self.base_path = c.get("BASE_PATH")
        self.downloaded_rawbook_dir = c.get("Downloaded_rawbook_dir")
        self.downloaded_data_meta = c.get("Downloaded_data_meta")
        self.story_input = c.get("Story_input")
        self.chroma_collection_name = (
            c.get("Chroma_collection_name_2") or c.get("Chroma_collection_name") or "English_Stories"
        )
        self.generated_story_output = c.get("Generated_story_output")

    def search_book(self, query: str, n_results: int = 20) -> dict:
        return parameters.search_books(self.config.get("Google_book_api_key"), query, n_results)

    def download_book(self, query: str, urls: list, formats: list) -> dict:
        return parameters.download_book_archive(
            self.base_path,
            query,
            formats,
            self.downloaded_rawbook_dir,
            self.downloaded_data_meta,
            self.config.get("Archive_url", ""),
        )

    def extract_text(self) -> dict:
        return parameters.extract_text_result()

    def reset_vector_store(self, *, new_collection_name: str = "StoryForgeRag_v1") -> dict:
        return parameters.reset_vector_store_result(new_collection_name=new_collection_name)

    def step1_prepare_and_enrich(
        self,
        *,
        limit: int = 0,
        overwrite_summary: bool = False,
        overwrite_sections: bool = False,
        dry_run: bool = False,
    ) -> dict:
        return parameters.step1_prepare_and_enrich_result(
            limit=limit,
            overwrite_summary=overwrite_summary,
            overwrite_sections=overwrite_sections,
            dry_run=dry_run,
        )

    def ingest_stories(self, *, collection_name: str = "StoryForgeRag_v1") -> dict:
        return parameters.ingest_stories_result(collection_name=collection_name)

    def query_vector_store(self, query: str, n_results: int = 5, collection: Optional[str] = None) -> dict:
        return parameters.query_vector_result(query, n_results)

    def vector_store_query(self, query: str, n_results: int = 5, collection: str = "") -> dict:
        return self.query_vector_store(query, n_results, collection or self.chroma_collection_name)

    def vector_store_insert(self, collection: str, ids: list[str], metadata: dict) -> dict:
        return parameters.vector_insert_result(ids, metadata)

    def vector_store_update(self, collection: str, ids: list[str], metadata: dict) -> dict:
        return parameters.vector_update_result(ids, metadata)

    def vector_store_delete(self, collection: str, ids: list[str], metadata: dict) -> dict:
        return parameters.vector_delete_result(ids)

    def generate_story(
        self,
        query: str,
        generation_type: str = "full_story",
        save: bool = True,
        n_results: int = 5,
        mode: Gen_mode = Gen_mode.FAST,
        story_type: StoryType = StoryType.MIX,
        debug: bool = False,
    ) -> dict:
        return parameters.generate_story_result(
            query,
            save,
            n_results,
            mode,
            story_type=story_type,
            debug=debug,
        )

    def generate_story_agentic(
        self,
        query: str,
        save: bool = True,
        mode: Gen_mode = Gen_mode.FAST,
        story_type: StoryType = StoryType.MIX,
        debug: bool = False,
    ) -> dict:
        return parameters.generate_story_agentic_result(
            query,
            save,
            mode,
            story_type=story_type,
            debug=debug,
        )

    def generate_summary(self, story_path: str, mode: Gen_mode = Gen_mode.FAST) -> dict:
        return parameters.generate_summary_result(
            story_path, self.base_path, self.config.get("Generated_summary_output", "Summarized_Stories")
        )

    def evaluate_story(self, story_text: str, context: Optional[str] = None, save: bool = False) -> dict:
        return parameters.evaluate_story_text_result(story_text)

    def evaluate_story_file(self, story_path: str, context_path: Optional[str] = None, save: bool = False) -> dict:
        return parameters.evaluate_story_file_result(story_path, save)

    def evaluate_summary(self, summary_path: str, context: Optional[str] = None, save: bool = False) -> dict:
        return parameters.evaluate_summary_result(summary_path, context, save)

    def run_pipeline(
        self,
        title: str,
        steps: Optional[list[str]] = None,
        gen_mode: Gen_mode = Gen_mode.FAST,
        story_type: StoryType = StoryType.MIX,
    ) -> dict:
        # Default pipeline: ingest stories → generate.
        # If Agentic_loop_enabled in setup.yaml, step 4 uses the evaluate/refine loop.
        agentic_enabled = str(self.config.get("Agentic_loop_enabled") or "").strip().lower() in ("true", "1", "yes")
        gen_step = "4_generate_story_agentic" if agentic_enabled else "4_generate_story_3step"
        default_steps = [
            "0_fetch_and_extract",
            "1_prepare_and_enrich_story_json",
            "2_reset_vector_store",
            "3_ingest_stories",
            gen_step,
        ]
        steps_to_run = steps or default_steps
        done: list[str] = []
        self._template_count = 0
        try:
            for step in steps_to_run:
                logging.info(f"Running step: {step}")
                if step == "0_fetch_and_extract":
                    res_download = self.download_book(query=title, urls=[], formats=["pdf", "epub"])
                    if not res_download.get("success"):
                        logging.warning(f"download: {res_download.get('message', res_download.get('error', 'failed'))}")
                    res_extract = self.extract_text()
                    if not res_extract.get("success"):
                        logging.warning(f"extract: {res_extract.get('message', res_extract.get('error', 'failed'))}")
                    done.append("0_fetch_and_extract")

                elif step == "1_prepare_and_enrich_story_json":
                    res_step1 = self.step1_prepare_and_enrich()
                    if not res_step1.get("success"):
                        raise RuntimeError(res_step1.get("error", "step1_prepare_and_enrich failed"))
                    done.append("1_prepare_and_enrich_story_json")

                elif step == "2_reset_vector_store":
                    res_reset = self.reset_vector_store(new_collection_name="StoryForgeRag_v1")
                    if not res_reset.get("success"):
                        raise RuntimeError(res_reset.get("error", "vector store reset failed"))
                    done.append("2_reset_vector_store")

                elif step == "3_ingest_stories":
                    res_ingest = self.ingest_stories(collection_name="StoryForgeRag_v1")
                    if not res_ingest.get("success"):
                        raise RuntimeError(res_ingest.get("error", "ingest_stories failed"))
                    logging.info("Ingested %s chunk(s) from Stories/*.txt", res_ingest.get("chunks_written", 0))
                    done.append("3_ingest_stories")

                elif step == "4_generate_story_3step":
                    _cfg = load_config(CONFIG_FILE)
                    n_results = _cfg.get("Story_generation_n_results", 3)
                    logging.info("Generate: using Story_generation_n_results=%s", n_results)
                    res = self.generate_story(
                        query=title,
                        generation_type="full_story",
                        save=True,
                        n_results=n_results,
                        mode=gen_mode,
                        story_type=story_type,
                    )
                    if not res.get("success"):
                        raise RuntimeError("generate failed")
                    done.append("4_generate_story_3step")

                elif step == "4_generate_story_agentic":
                    res = self.generate_story_agentic(
                        query=title,
                        save=True,
                        mode=gen_mode,
                        story_type=story_type,
                    )
                    if not res.get("success"):
                        raise RuntimeError("agentic generate failed")
                    logging.info(
                        "Agentic generate: stop_reason=%s iterations=%s avg=%s",
                        res.get("stop_reason"),
                        res.get("iterations_run"),
                        res.get("final_average"),
                    )
                    done.append("4_generate_story_agentic")

                else:
                    raise ValueError(f"Unknown step: {step}")
            return {"success": True, "message": "Pipeline completed", "steps_done": done, "error": None}
        except Exception as e:
            logging.exception("Pipeline failed")
            return {"success": False, "message": "Pipeline failed", "steps_done": done, "error": str(e)}
