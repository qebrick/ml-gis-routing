"""Download Porto's driving network with Overpass fallback endpoints."""

from pathlib import Path
import time

import osmnx as ox

OUT = Path("data/graphs/porto_drive.graphml")
PLACE = "Porto, Portugal"

OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api",
    "https://overpass-api.de/api",
    "https://overpass.nchc.org.tw/api",
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем ответы в кэш, чтобы повторно не запрашивать
    # уже загруженные данные.
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 180

    last_error = None
    G = None

    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = endpoint
        print(f"\nTrying Overpass server: {endpoint}")

        for attempt in range(1, 3):
            try:
                G = ox.graph_from_place(
                    PLACE,
                    network_type="drive",
                    simplify=True,
                )
                print(f"Download succeeded using {endpoint}")
                break

            except Exception as exc:
                last_error = exc
                print(
                    f"Attempt {attempt}/2 failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt < 2:
                    time.sleep(5)

        if G is not None:
            break

    if G is None:
        raise RuntimeError(
            "Could not download the road graph from any Overpass "
            "endpoint. Check network access and try again later."
        ) from last_error

    G = ox.distance.add_edge_lengths(G)
    ox.save_graphml(G, OUT)

    print(f"\nSaved graph: {OUT}")
    print(f"Nodes: {len(G):,}")
    print(f"Directed edges: {G.number_of_edges():,}")


if __name__ == "__main__":
    main()