from ultralytics.data.converter import convert_coco

for i in [10, 25, 50, 100]:
    for j in [1, 5]:
        src = f'gen{i}_{j}shot'
        dst = f'distilled_datasets/{j}r{i}g/gen'
        convert_coco(src, dst)