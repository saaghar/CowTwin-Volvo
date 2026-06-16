from ultralytics.data.split import autosplit

autosplit(
    path="distill_dataset/images",
    weights=(0.9, 0.1, 0.0),
    annotated_only=False,
)