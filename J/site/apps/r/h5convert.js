// juxtaposed app: read 10x HDF5 count matrices in the browser with h5wasm, then hand them to R.
// The R build in the browser (webR) cannot read HDF5, so .h5 files are converted here into a small
// binary file (".jx10x") that R reads with readBin(). Other files pass through unchanged.
//
// Binary layout (little endian):
//   "JX10X1\n\0" (8 bytes) | uint32 header length | header JSON (UTF-8) | padding to 4 bytes |
//   int32 indptr[n_indptr] | int32 indices[nnz] | data[nnz] as int32 ("int") or float64 ("float")
// Header JSON: {shape:[n_features,n_barcodes], nnz, n_indptr, data_type, barcodes:[], names:[], ids:[], types:[], source}

const H5WASM = "https://cdn.jsdelivr.net/npm/h5wasm@0.10.3/dist/esm/hdf5_hl.js";
let h5mod = null;
let h5FS = null;

async function h5() {
  if (!h5mod) {
    const m = await import(H5WASM);
    h5mod = m.default;
    const module = await h5mod.ready;
    h5FS = module.FS;
  }
  return h5mod;
}

function toInt32(a) {
  if (a instanceof Int32Array) return a;
  if (a instanceof BigInt64Array || a instanceof BigUint64Array) return Int32Array.from(a, Number);
  return Int32Array.from(a);
}

function strings(v) {
  if (v == null) return [];
  return Array.from(v, (s) => (typeof s === "string" ? s : String(s)));
}

export async function convert10xH5(file, onStatus) {
  const lib = await h5();
  const FS = h5FS;
  const tmp = "/upload_" + Date.now() + "_" + Math.floor(Math.random() * 1e6) + ".h5";
  onStatus && onStatus("Reading " + file.name + " …");
  FS.writeFile(tmp, new Uint8Array(await file.arrayBuffer()));
  const f = new lib.File(tmp, "r");
  try {
    let g = f.get("matrix");
    let names, ids, types, source;
    if (g && g.keys().includes("data")) {
      const feat = g.keys().includes("features") ? g.get("features") : null;
      if (feat) {
        names = strings(feat.get("name").value);
        ids = feat.keys().includes("id") ? strings(feat.get("id").value) : names;
        types = feat.keys().includes("feature_type") ? strings(feat.get("feature_type").value) : names.map(() => "Gene Expression");
      } else {
        names = strings(g.get("gene_names").value);
        ids = strings(g.get("genes").value);
        types = names.map(() => "Gene Expression");
      }
      source = "10x HDF5 (v3)";
    } else {
      // Cell Ranger 2 layout: one group per genome with gene_names / genes
      const k = f.keys().find((key) => {
        const x = f.get(key);
        return x && typeof x.keys === "function" && x.keys().includes("data");
      });
      if (!k) throw new Error(file.name + " is not a 10x count matrix (no matrix/data dataset).");
      g = f.get(k);
      names = strings(g.get("gene_names").value);
      ids = strings(g.get("genes").value);
      types = names.map(() => "Gene Expression");
      source = "10x HDF5 (v2, genome " + k + ")";
    }
    const shape = Array.from(g.get("shape").value, Number);
    const indptr = toInt32(g.get("indptr").value);
    const indices = toInt32(g.get("indices").value);
    const raw = g.get("data").value;
    const isInt = raw instanceof Int32Array || raw instanceof Uint32Array || raw instanceof Int16Array ||
      raw instanceof Uint16Array || raw instanceof BigInt64Array || raw instanceof BigUint64Array;
    const data = isInt ? toInt32(raw) : Float64Array.from(raw);
    const barcodes = strings(g.get("barcodes").value);
    const header = {
      shape, nnz: indices.length, n_indptr: indptr.length, data_type: isInt ? "int" : "float",
      barcodes, names, ids, types, source, file: file.name,
    };
    const hjson = new TextEncoder().encode(JSON.stringify(header));
    const magic = new TextEncoder().encode("JX10X1\n\0");
    const len = new Uint8Array(new Uint32Array([hjson.length]).buffer);
    const pad = new Uint8Array((4 - ((8 + 4 + hjson.length) % 4)) % 4);
    onStatus && onStatus("Converted " + file.name + ": " + shape[0] + " features × " + shape[1] + " barcodes");
    const base = file.name.replace(/\.(h5|hdf5)$/i, "");
    return new File([magic, len, hjson, pad, indptr, indices, data], base + ".jx10x",
      { type: "application/octet-stream" });
  } finally {
    f.close();
    try { FS.unlink(tmp); } catch (e) { /* ignore */ }
  }
}

// Wire a visible picker (#jx_picker) to the hidden Shiny fileInput (#upload). The picker is created
// later by Shiny (renderUI), so listen on the document instead of on the element.
export function wirePicker(pickerId, shinyInputId, statusId) {
  const say = (msg, isError) => {
    const status = document.getElementById(statusId);
    if (!status) return;
    status.textContent = msg;
    status.className = "jx-status" + (isError ? " jx-error" : "");
  };
  document.addEventListener("change", async (event) => {
    const picker = event.target;
    if (!picker || picker.id !== pickerId) return;
    const files = Array.from(picker.files || []);
    if (!files.length) return;
    const out = [];
    try {
      for (const file of files) {
        if (/\.(h5|hdf5)$/i.test(file.name)) out.push(await convert10xH5(file, (m) => say(m)));
        else out.push(file);
      }
    } catch (e) {
      say("Could not read the HDF5 file: " + e.message, true);
      picker.value = "";
      return;
    }
    const target = document.getElementById(shinyInputId);
    if (!target) { say("Internal error: upload field not found.", true); return; }
    const dt = new DataTransfer();
    out.forEach((x) => dt.items.add(x));
    target.files = dt.files;
    target.dispatchEvent(new Event("change", { bubbles: true }));
    say("Read " + out.map((x) => x.name).join(", ") + ". Add more files or press Load data.");
    picker.value = "";
  });
}
