#!/bin/zsh

trap 'on_exit' SIGINT

on_exit() {
    rm -rf figures_*
    rm -rf pdfs
    rm -rf lancedb
    mkdir pdfs
    exit 0
}

streamlit run landing_page.py &
wait $!
