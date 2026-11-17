import json, sys
nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    print(f"--- [{i}] {c['cell_type']} ---")
    print(src)
    if c["cell_type"] == "code":
        for o in c.get("outputs", []):
            t = o.get("output_type")
            if t == "stream":
                txt = "".join(o.get("text", ""))
                print("  >> " + txt[:600].replace("\n", "\n  >> "))
            elif t in ("execute_result", "display_data"):
                d = o.get("data", {})
                if "text/plain" in d:
                    print("  => " + "".join(d["text/plain"])[:600].replace("\n", "\n  => "))
                if "image/png" in d:
                    print("  => [image]")
            elif t == "error":
                print("  !! " + o.get("ename", "") + ": " + o.get("evalue", "")[:200])
