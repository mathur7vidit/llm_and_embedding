# llm_and_embedding

A collection of Jupyter notebooks demonstrating experiments with large language models (LLMs), text summarization and generation, sentiment analysis, and embedding / similarity techniques. The notebooks use Hugging Face Transformers, Gensim (word embeddings & soft-cosine similarity), and common Python data science libraries.

Repository: https://github.com/mathur7vidit/llm_and_embedding

## Contents

- `week_6_section_a.ipynb` — Text generation experiments (GPT-2 and other causal LM examples). Shows generation pipelines, cleaning/sanitizing completions, and prompt handling.
- `week_6_section_b.ipynb` — Summarization and generation experiments (BART summarizer, Qwen causal LM examples, a small poem generator, and utilities to generate concise clean summaries).
- `week_6_section_c.ipynb` — Embeddings and similarity experiments (Word2Vec examples, SoftCosine similarity, similarity indices and visualizations).
- `week_6_section_d.ipynb` — Simple Hugging Face pipeline demo for sentiment analysis.

## Quick overview

These notebooks illustrate typical workflows:
- Load tokenizer and model(s) via Hugging Face Transformers.
- Use `pipeline` for summarization, generation, and sentiment analysis.
- Build word embeddings (Word2Vec) and compute SoftCosine similarity using Gensim.
- Short helper code to post-process model outputs (e.g., limit summaries to ≤ 30 words, clean completions to end at sentence boundaries).

## Requirements

Recommended Python version: 3.8+

Minimum packages (suggested):
- jupyterlab or notebook
- transformers
- torch (or `accelerate` if using device offloading)
- gensim
- numpy
- matplotlib
- pandas (optional)
- scikit-learn (optional)

Example install (adjust versions as needed, and prefer a virtual environment):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyterlab transformers torch gensim numpy matplotlib pandas
```

You can also create a `requirements.txt` with the above packages for reproducibility.

## Running the notebooks

1. Start Jupyter:
   ```bash
   jupyter lab
   ```
2. Open any `week_6_section_*.ipynb` notebook and run cells in order.
3. Notes:
   - Some models are large; running them on CPU may be slow. If available, configure GPU device (PyTorch + CUDA).
   - The notebooks sometimes set `device=-1` (CPU) in `pipeline` calls; change to appropriate device index for GPU (e.g., `device=0`) or use `device_map="auto"` for large models with `from_pretrained`.
   - For Hugging Face Hub models that require authentication or rate limits, set `HF_HOME`/`HF_TOKEN` or run `huggingface-cli login`.

## Tips & environment variables
- The notebooks set `os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"` in some places to suppress symlink warnings when loading models — this is optional.
- For very large models, consider `transformers` configuration that supports model offloading (e.g., `device_map="auto"` and `torch_dtype="auto"`).

## Reproducibility & data
- Notebooks are primarily demo/teaching artifacts — they include inline example texts and small synthetic inputs. No external dataset files are required.
- If you want to reproduce precise outputs, pin package versions (especially `transformers` and `torch`).

## Security & model safety
- Model-generated outputs are not filtered for safety. Use appropriate moderation or safety checks before using generated text in production.
- When running third-party or community models from the Hugging Face Hub, review the model card for license and usage notes.

## Suggested next steps / improvements
- Add a `requirements.txt` or `environment.yml` for exact reproducibility.
- Add small README sections per notebook with a one-line run-through and expected outputs.
- Provide a tiny CLI or script wrapper to run the summarization demo non-interactively (e.g., `python demos/summarize.py`).
- Save model outputs/artifacts (e.g., embeddings) to disk and include a small example of loading them back to avoid repeated downloads.

## Contributing
Contributions are welcome. If you want to add examples, fixes, or another notebook:
1. Fork the repository.
2. Create a branch for your change.
3. Open a pull request with a brief description of what you changed and why.

## License
Add a license of your choice (e.g., MIT) by including a `LICENSE` file. If you want, I can add a suggested `LICENSE` file too.

## Contact
Repo owner: @mathur7vidit (GitHub)
