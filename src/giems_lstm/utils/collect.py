import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from giems_lstm.config import Config


def _collect(config_path: str, eval: bool, parallel: int):
    from giems_lstm.engine import Collector

    logger = logging.getLogger()
    config = Config(config_path=config_path, mode="analyze")
    collector = Collector(config=config, eval=eval)
    files = list(collector.source_folder.glob("*.npy"))
    chunk_size = config.sys.tasks_per_thread
    batches = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
    total_loaded = 0

    if parallel <= 1:
        for batch in batches:
            try:
                collector.run(batch)
                total_loaded += len(batch)
                logger.debug(f"Loaded {total_loaded}/{config.total_tasks} files...")
            except Exception as e:
                logger.error(f"Error loading batch: {e}")
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_batch = {
                executor.submit(collector.run, batch): batch for batch in batches
            }

            for future in as_completed(future_to_batch):
                try:
                    future.result()
                    total_loaded += len(future_to_batch[future])
                    logger.debug(f"Loaded {total_loaded}/{config.total_tasks} files...")
                except Exception as e:
                    logger.error(f"Error loading batch: {e}")

    collector.save()
    logger.info("Collection and saving complete.")
