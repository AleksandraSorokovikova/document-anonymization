# Evaluation Report

## Model and experiment information

Model: YOLOv10-Document-Layout-Analysis

Image size: 960x960

Epochs: 100

Dataset size: 2k images



### Results

| Class              |   Images |   Instances |   Precision |    Recall |    mAP50 |
|:-------------------|---------:|------------:|------------:|----------:|---------:|
| first_name         |       59 |         174 |    0.354839 | 0.0632184 | 0.354839 |
| last_name          |       67 |         327 |    0.3      | 0.0550459 | 0.3      |
| address            |       59 |         175 |    0.347368 | 0.188571  | 0.347368 |
| phone_number       |       33 |          80 |    0.461538 | 0.3       | 0.461538 |
| email_address      |        7 |          16 |    0.285714 | 0.375     | 0.285714 |
| dates              |       84 |         168 |    0.368421 | 0.208333  | 0.368421 |
| credit_card_number |        0 |           0 |    0        | 0         | 0        |
| iban               |        0 |           0 |    0        | 0         | 0        |
| company_name       |       63 |         130 |    0.272727 | 0.115385  | 0.272727 |
| signature          |       39 |          50 |    0.25     | 0.26      | 0.25     |
| full_name          |       68 |         287 |    0.352941 | 0.0418118 | 0.352941 |
| vin                |        0 |           0 |    0        | 0         | 0        |
| car_plate          |        0 |           0 |    0        | 0         | 0        |
| all                |       89 |        1407 |    0.291449 | 0.118692  | 0.230273 |

---

