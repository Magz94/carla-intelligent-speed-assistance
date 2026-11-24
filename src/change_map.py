#!/usr/bin/env python3

import argparse
import carla

def main():
    parser = argparse.ArgumentParser(description="Load a CARLA town")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("-p", "--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "-t", "--town", default="Town02",
        help="Name of the town to load (e.g. Town02, Town04, Town10HD, etc.)"
    )
    args = parser.parse_args()

    # 1) Connect to the CARLA server
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    # 2) Load the specified town
    print(f"Loading town '{args.town}' ...")
    world = client.load_world(args.town)

    print("Town loaded")

if __name__ == "__main__":
    main()
