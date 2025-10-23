type Event = {
  event_id: string;
  event_type: string;
  timestamp: string;
  session_id: string;
  user_id: string;
  region: string;
  properties?: Record<string, any>;
};

function sendEvent(event: Event) {
  const url = "http://localhost:8000/collect";
  const payload = JSON.stringify(event);
  if (navigator.sendBeacon) {
    const blob = new Blob([payload], { type: "application/json" });
    navigator.sendBeacon(url, blob);
  } else {
    fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: payload }).catch(() => {});
  }
}

export { sendEvent };
