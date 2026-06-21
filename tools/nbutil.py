"""
nbutil.py  -  a tiny helper for building Jupyter notebooks from plain Python.

Why this exists
---------------
A .ipynb file is just JSON. Hand-writing that JSON is error prone, so every
chapter in this course is generated from a small, readable Python script that
calls the three helpers below:

    md(text)   -> a Markdown cell  (explanations, equations in LaTeX, headings)
    code(text) -> a Code cell      (Python the student runs)
    save(path, cells) -> writes the .ipynb to disk

This keeps the *source of truth* as clean Python text instead of messy JSON,
and lets us regenerate every notebook reproducibly.

LaTeX note: inside md(...) you can write math with $ ... $ (inline) or
$$ ... $$ (display). Jupyter renders it automatically.
"""

import json
import os

_counter = [0]


def _next_id():
    """Deterministic, unique cell ids (nbformat 4.5+ wants them)."""
    _counter[0] += 1
    return "cell-%04d" % _counter[0]


def _src(text):
    """Turn a Python string into the list-of-lines format nbformat expects."""
    text = text.strip("\n")
    return text.splitlines(keepends=True)


def md(text):
    """Create a Markdown cell."""
    return {"cell_type": "markdown", "id": _next_id(),
            "metadata": {}, "source": _src(text)}


def code(text):
    """Create a Code cell (starts with no output; the student runs it)."""
    return {
        "cell_type": "code",
        "id": _next_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(text),
    }


def save(path, cells):
    """Assemble cells into a valid notebook and write it to `path`."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print("wrote", path, "(%d cells)" % len(cells))
