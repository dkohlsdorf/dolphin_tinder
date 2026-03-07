export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/^\//, ""); // Clean the path

    // 1. Handle API/Data requests from your React Frontend
    if (path.startsWith("data/") || path.startsWith("pairs/")) {
      const rangeHeader = request.headers.get("Range");

      // First get object metadata to know the size
      const headObject = await env.BUCKET.head(path);
      if (!headObject) {
        return new Response("Dolphin file not found", { status: 404 });
      }

      const totalSize = headObject.size;
      const headers = new Headers();
      headObject.writeHttpMetadata(headers);
      headers.set("Access-Control-Allow-Origin", "*");
      headers.set("etag", headObject.httpEtag);
      headers.set("Accept-Ranges", "bytes");

      // Handle range requests for audio/video playback
      if (rangeHeader) {
        const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
        if (match) {
          const start = parseInt(match[1], 10);
          const end = match[2] ? parseInt(match[2], 10) : totalSize - 1;

          // Validate range
          if (start >= totalSize || end >= totalSize || start > end) {
            return new Response("Range Not Satisfiable", {
              status: 416,
              headers: { "Content-Range": `bytes */${totalSize}` }
            });
          }

          const object = await env.BUCKET.get(path, {
            range: { offset: start, length: end - start + 1 }
          });

          headers.set("Content-Range", `bytes ${start}-${end}/${totalSize}`);
          headers.set("Content-Length", end - start + 1);

          return new Response(object.body, { status: 206, headers });
        }
      }

      // Full request (no range)
      const object = await env.BUCKET.get(path);
      headers.set("Content-Length", totalSize);
      return new Response(object.body, { headers });
    }

    // 2. Serve your index.html for everything else
    // This uses the 'assets' binding from your wrangler.toml
    return env.ASSETS.fetch(request);
  },
};
