# Конвейер статьи

Область проекта: «Адаптивный выбор ориентиров ALT в алгоритме A* на дорожном графе Порту: сравнение дистанционного правила и дерева решений». Основная стоимость — `osm_travel_time_s`, даже когда граф называется `conservative_refit`.

## Опубликованные результаты

| Часть статьи | Данные | Код |
|---|---|---|
| Основной эксперимент, таблицы 6–9 | `porto_heldout_alt_benchmark.parquet` | `benchmark_porto_heldout_alt`, `finalize_porto_alt_results`, `analyze_porto_heldout_alt` |
| Дальность и устойчивость, таблицы 10–11 | тот же benchmark | `audit_porto_alt_robustness` |
| Состав нагрузки, таблица 12 | benchmark, `porto_heldout_600_od.parquet`, `ml_alt_budget_groupkfold.parquet` | `analyze_porto_alt_workload` |
| Фиксированные 600 OD | `porto_heldout_600_od.parquet` | `create_heldout_porto_od` |
| 142 GPS-пары | `gps_od_route_comparison.parquet` | `compare_gps_od_routes` |
| Набор 32 ориентиров | `adaptive_alt_landmarks.csv` | `benchmark_adaptive_alt_k` |
| Признаки и обучающие цели дерева | `ml_alt_budget_groupkfold.parquet` | `validate_ml_alt_budget` |
| Отдельная калибровка, таблица 2 | `conservative_refit_coefficients.json` | `build_conservative_refit_graph` |

Имена модулей в таблице приведены без `src.routing.`. Основные команды:

```bash
python -m src.article verify
python -m src.article reports
python -m src.article benchmark --output-dir reports/run-001
python -m src.article reports --benchmark reports/run-001/porto_heldout_alt_benchmark.parquet --output-dir reports/run-001-analysis
```

Все опубликованные данные остаются в `data/`; новые результаты этих команд — в `reports/`. Файл `porto_alt_recalculation.json` описывает среду пересчёта и SHA-256 входного benchmark, а `data/processed/porto_alt_reproducibility.json` хранит исходные метаданные. Прямые команды пяти основных модулей используют тот же безопасный каталог вывода; для защиты от повторного замера поверх существующего файла используйте `src.article benchmark`.

## Происхождение подготовленных данных

Ниже — порядок зависимостей сохранённых исследовательских сценариев. Это не обещание побитового восстановления первичных данных: исходный `train.csv` и точная история его отбора отсутствуют, а повторное скачивание OSM даёт актуальный граф. Сценарии подготовки запускаются из корня и **перезаписывают** свои промежуточные файлы; для нового исследования используйте отдельную копию проекта.

1. `src.data.download_graph` → `porto_drive.graphml`.
2. `src.data.prepare_trajectories --input data/raw/train.csv` → `gps_intervals.parquet`.
3. `src.data.map_matching_pilot` → `src.data.clean_map_matching` → `src.data.fix_segments` → `map_matching_segments_v2.parquet`.
4. `src.data.hmm_map_matching` → `hmm_pilot.parquet`; `src.data.check_hmm_interval_quality` и `src.data.extract_hmm_routes` → качество интервалов и пройденные дуги.
5. `src.routing.build_osm_weights` → `src.routing.prepare_routing_graph` → граф с положительными OSM-весами.
6. `src.ml.prepare_interval_training_data` → `src.ml.validate_osm_intervals` → `src.ml.detect_hmm_backtracking` → `src.ml.build_distance_based_routes` → `src.ml.prepare_road_class_time_features`.
7. `src.routing.build_class_weighted_graph` → `src.routing.build_conservative_refit_graph`. `src.ml.validate_conservative_refit` сохраняет проверку калибровки. Эти данные нужны для таблицы 2, хотя калиброванный вес не является целевой функцией ALT.
8. `src.routing.compare_gps_od_routes` → исходные GPS-пары.
9. `src.routing.benchmark_adaptive_alt_k` → набор ориентиров; `src.routing.benchmark_adaptive_alt_end_to_end` → измерения на обучающих GPS-парах.
10. `src.routing.validate_ml_alt_budget` → таблица признаков и целей; `src.routing.validate_ml_alt_compact_tree`, `src.routing.benchmark_ml_alt_direct_tree`, `src.routing.benchmark_ml_alt_compiled_rule` → проверка компактного дерева и стоимости исполнения.
11. `src.routing.create_heldout_porto_od` → фиксированные 600 OD. Для воспроизведения основной статьи используйте уже опубликованный набор и проверяйте его SHA-256.

Сохранена также ветка разработки дистанционного правила: `diagnose_adaptive_alt_landmarks` → `analyze_alt_ml_opportunity` → `analyze_alt_presearch_features`, совместно с `benchmark_adaptive_alt_precomputed` → `validate_alt_distance_gate`.

## Границы реорганизации

Исследовательские сценарии подготовки остались исполняемыми модулями с операциями на верхнем уровне: импортировать их как библиотеку нельзя. Они запускаются через `python -m ...`. Новый интерфейс `src.article`, проверки `src.validation` и пути `src.paths` безопасны для импорта. Измеряемый алгоритм основного benchmark не переписывался, чтобы отделить организационные изменения от изменений вычислительного эксперимента.

Для таблицы 8 используйте `finalize_porto_alt_results`: диагностический `analyze_porto_heldout_alt` применяет другую последовательность случайных bootstrap-выборок, поэтому его границы интервалов могут немного отличаться.
