export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/^\//, ""); // Clean the path

    // 1. Handle API/Data requests from your React Frontend
    if (path.startsWith("data/") || path.startsWith("pairs/")) {
      const object = await env.BUCKET.get(path);

      if (!object) {
        return new Response("Dolphin file not found", { status: 404 });
      }

      // Prepare headers (Type, CORS, etc.)
      const headers = new Headers();
      object.writeHttpMetadata(headers);
      headers.set("Access-Control-Allow-Origin", "*"); 
      headers.set("etag", object.httpEtag);

      return new Response(object.body, { headers });
    }

    // 2. Serve your index.html for everything else
    // This uses the 'assets' binding from your wrangler.toml
    return env.ASSETS.fetch(request);
  },
};
