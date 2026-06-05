export async function loadSiteData(route = "#/") {
  const [latest, sourceStatus, pickem] = await Promise.all([
    getJson("./data/latest.json"),
    getJson("./data/system/source-status.json"),
    getJson("./data/pickem/current.json")
  ]);
  const stage = await getJson(`./data/stages/${stageIdFromRoute(route, latest.current_stage)}.json`);
  return { latest, sourceStatus, pickem, stage, route };
}

export async function loadStage(stageNumber) {
  return getJson(`./data/stages/stage-${stageNumber}.json`);
}

async function getJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
}

function stageIdFromRoute(route, currentStage) {
  const match = String(route || "").match(/^#\/stage\/([123])$/);
  if (match) {
    return `stage-${match[1]}`;
  }
  return currentStage;
}
