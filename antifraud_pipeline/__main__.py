# -*- coding: utf-8 -*-
"""Точка входа: ``python -m antifraud_pipeline`` из корня репозитория."""
from .antifraud_stages import run_all_stages

if __name__ == "__main__":
    run_all_stages()
