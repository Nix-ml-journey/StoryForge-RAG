import logging 
import yaml 
from pathlib import Path
from typing import Optional 
from Generative_AI.generative_ai import Gen_mode
from . import response_parameter as parameters 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT_DIR = Path(__file__).parent.parent
CONFIG_FILE = ROOT_DIR / "setup.yaml"

class Orchestrator:
    def __init__(self, config_path: Optional[str] = None):
        path = Path(config_path) if config_path else CONFIG_FILE
        self.config = yaml.safe_load(path.read_text(encoding="utf-8"))
        c = self.config
        self.base_path = c.get("BASE_PATH")
        self.downloaded_rawbook_dir = c.get("Downloaded_rawbook_dir")
        self.downloaded_data_meta = c.get("Downloaded_data_meta")
        self.extracted_metadata_dir = c.get("Extracted_metadata_dir")
        self.metadata_input = c.get("Metadata_input")
        self.story_input = c.get("Story_input")
        self.merged_data_output = c.get("Merged_data_output")
        self.chroma_collection_name = c.get("Chroma_collection_name")
        self.generated_story_output = c.get("Generated_story_output")

    def search_book(self, query: str, n_results: int = 20) -> dict:
        return parameters.search_books(self.config.get("Google_book_api_key"), query, n_results)

    def download_book(self, query: str, urls: list, formats: list) -> dict:
        return parameters.download_book_archive(self.base_path, query, formats, self.downloaded_rawbook_dir, self.downloaded_data_meta, self.config.get("Archive_url", ""))

    def extract_text(self) -> dict:
        return parameters.extract_text_result()

    def metadata_check(self, folder_path: str, file_name: str, number_books: int, empty_number: int) -> dict:
        return parameters.metadata_check_result(folder_path, file_name, number_books, empty_number)

    def metadata_update(self, metadata: dict, folder_path: str, file_name: str) -> dict:
        return parameters.metadata_update_result(metadata, folder_path, file_name)

    def create_metadata_template(self, number_books: int = 0, empty_number: int = 0) -> dict:
        return parameters.create_metadata_template_result(number_books, empty_number)

    def merge_data(self, metadata_input: Optional[str] = None, story_input: Optional[str] = None, merged_output: Optional[str] = None) -> dict:
        return parameters.merge_data_result(self.base_path, story_input or self.story_input, metadata_input or self.metadata_input, merged_output or self.merged_data_output)

    def create_summaries(self, merged_output: Optional[str] = None) -> dict:
        return parameters.create_summaries_result(self.base_path, merged_output or self.merged_data_output)

    def create_summaries_and_check(self) -> dict:
        return parameters.create_summaries_and_check_result()    

    def ingest(self, merged_output: Optional[str] = None) -> dict:
        return parameters.ingest_result(self.base_path, merged_output or self.merged_data_output)

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
        self, query: str, generation_type: str = "full_story", save: bool = True, n_results: int = 5, mode: Gen_mode = Gen_mode.FAST) -> dict:
        return parameters.generate_story_result(query, save, n_results, mode)

    def generate_summary(self, story_path: str, mode: Gen_mode = Gen_mode.FAST) -> dict:
        return parameters.generate_summary_result(story_path, self.base_path, self.config.get("Generated_summary_output", "Summarized_Stories"))

    def evaluate_story(self, story_text: str, context: Optional[str] = None, save: bool = False) -> dict:
        return parameters.evaluate_story_text_result(story_text)

    def evaluate_story_file(self, story_path: str, context_path: Optional[str] = None, save: bool = False) -> dict:
        return parameters.evaluate_story_file_result(story_path, save)

    def evaluate_summary(self, summary_path: str, context: Optional[str] = None, save: bool = False) -> dict:
        return parameters.evaluate_summary_result(summary_path, context, save) 

    def run_pipeline(self, title: str, steps: Optional[list[str]] = None, gen_mode: Gen_mode = Gen_mode.FAST) -> dict:
        default_steps = [
            "1_fetch_and_extract",
            "2_create_metadata_template",
            "3_merge_check_summary",
            "4_ingest",
            "5_generate_and_evaluate",
        ]
        steps_to_run = steps or default_steps
        done: list[str] = []
        self._template_count = 0
        try:
            for step in steps_to_run:
                logging.info(f"Running step: {step}")
                if step == "1_fetch_and_extract":
                    res_download = self.download_book(query=title, urls=[], formats=["pdf", "epub"])
                    if not res_download.get("success"):
                        logging.warning(f"download: {res_download.get('message', res_download.get('error', 'failed'))}")
                    res_extract = self.extract_text()
                    if not res_extract.get("success"):
                        logging.warning(f"extract: {res_extract.get('message', res_extract.get('error', 'failed'))}")
                    done.append("1_fetch_and_extract")

                elif step == "2_create_metadata_template":
                    res = self.create_metadata_template()
                    if not res.get("success"):
                        logging.warning(f"create_metadata_template: {res.get('message', 'failed')}")
                    self._template_count = res.get("template_count", 0)
                    logging.info(f"Metadata templates created: {self._template_count}")
                    done.append("2_create_metadata_template")

                elif step == "3_merge_check_summary":
                    res_merge = self.merge_data()
                    if not res_merge.get("success"):
                        raise RuntimeError("merge failed")
                    res_check = self.metadata_check(
                        self.base_path, self.metadata_input, 0, 0
                    )
                    if not res_check.get("success"):
                        logging.warning(f"Empty summaries in Metadata: {res_check.get('issues', res_check.get('empty_number', 0))} files")
                    res_summary = parameters.create_summaries_and_check_result()
                    empty_count = res_summary.get("empty_count", 0)
                    created_count = res_summary.get("summaries_created", res_summary.get("created_count", 0))
                    counts_match = res_summary.get("counts_match", False)
                    template_count = getattr(self, "_template_count", 0)
                    if not res_summary.get("success"):
                        logging.warning(f"create_summaries_and_check: {res_summary.get('error', 'failed')}")
                    if template_count != created_count:
                        logging.warning(
                            f"Template vs created summary count: template_count={template_count}, created_count={created_count} (should be same)"
                        )
                    elif not counts_match:
                        logging.warning(f"check_number_summaries mismatch: empty_count={empty_count}, created_count={created_count}")
                    else:
                        logging.info(f"Template and summary counts match: template_count={template_count}, created_count={created_count}")
                    done.append("3_merge_check_summary")

                elif step == "4_ingest":
                    res_ingest = self.ingest()
                    ingest_count = res_ingest.get("count", 0)
                    logging.info(f"Ingested {ingest_count} file(s) into the vector store")
                    if not res_ingest.get("success"):
                        logging.warning(f"ingest: {res_ingest.get('error', 'failed')}")
                    done.append("4_ingest")

                elif step == "5_generate_and_evaluate":
                    _cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
                    n_results = _cfg.get("Story_generation_n_results", 3)
                    logging.info(f"Step 5: using Story_generation_n_results={n_results} from setup.yaml")
                    res = self.generate_story(query=title, generation_type="full_story", save=True, n_results=n_results, mode=gen_mode)
                    if not res.get("success"):
                        raise RuntimeError("generate failed")
                    try:
                        story_dir = Path(self.base_path) / self.generated_story_output
                        if story_dir.exists():
                            for p in sorted(story_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)[:1]:
                                self.evaluate_story_file(str(p), save=True)
                                break
                    except Exception as e:
                        logging.warning(f"evaluate: {e}")
                    done.append("5_generate_and_evaluate")

                else:
                    raise ValueError(f"Unknown step: {step}")
            return {"success": True, "message": "Pipeline completed", "steps_done": done, "error": None}
        except Exception as e:
            logging.exception("Pipeline failed")
            return {"success": False, "message": "Pipeline failed", "steps_done": done, "error": str(e)}