export default function DemoNotice({ label }: { label: string }) {
  return (
    <div
      className="container"
      style={{
        paddingTop: 20,
        paddingBottom: 0,
      }}
    >
      <div
        className="meta-text"
        style={{
          border: "0.5px solid var(--watch)",
          color: "var(--watch)",
          padding: "10px 12px",
          background: "rgba(217, 119, 6, 0.08)",
        }}
      >
        Development demo: {label}. This route is isolated from the production
        dashboard workflow.
      </div>
    </div>
  );
}
