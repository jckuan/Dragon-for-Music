def clean_environment_file(input_file="env-mmmr-raw.yml", output_file="env-mmmr.yml"):
    with open(input_file, "r") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        if line.strip().startswith("-"):
            pkg = line.strip()[2:].split("=")[0]

            # Heuristic: drop low-level system packages
            if pkg.startswith("lib") or pkg in {
                "ncurses", "readline", "xz", "zlib", "tk", "openssl", "ca-certificates"
            }:
                continue

        cleaned.append(line)

    with open(output_file, "w") as f:
        f.writelines(cleaned)

    print(f"✅ Cleaned environment written to {output_file}")


if __name__ == "__main__":
    clean_environment_file()
