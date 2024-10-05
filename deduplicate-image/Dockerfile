FROM python:3

WORKDIR /usr/src/app

COPY . .

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN pip install numpy scipy sentencepiece text-dedup

CMD [ "bash" ]
# CMD [ "bash", "python -m text_dedup.suffix_array --path \"allenai/c4\" --name \"lt\" --split \"train\" --cache_dir \"./cache\" --output \"output\" --column \"text\" --google_repo_path \"deduplicate-text-datasets\"" ]
