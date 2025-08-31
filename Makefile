DATA=/data
OUT=/app/output
MANIFEST=$(OUT)/manifest/episodes.parquet
VALID=$(OUT)/validation
STATS=$(OUT)/stats/global_stats.json
NORM=$(OUT)/normalized
DS=$(OUT)/dataset

.PHONY: pipeline discover validate stats align materialize clean

pipeline: discover validate stats align materialize

discover:
	docker compose run --rm neura \
	  discover --data-root $(DATA) --manifest $(MANIFEST)

validate:
	docker compose run --rm neura-media \
	  validate --manifest $(MANIFEST) --meta-dir $(DATA)/meta --out $(VALID) --skip-video

stats:
	docker compose run --rm neura \
	  stats --data-root $(DATA) --validated-ids $(VALID)/validated_episodes.jsonl --out $(STATS)

align:
	docker compose run --rm neura \
	  align-transform --data-root $(DATA) --out $(NORM) --stats $(STATS)

materialize:
	docker compose run --rm neura \
	  materialize --norm-dir $(NORM) --out $(DS) --videos-root /app/robot_data/videos --link-videos symlink

clean:
	rm -rf output
