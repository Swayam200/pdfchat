'use client';
import { ReactNode } from "react";
import { Panel, Group as PanelGroup, Separator as PanelResizeHandle } from "react-resizable-panels";

interface ResizablePanesProps {
  left: ReactNode;
  right: ReactNode;
}

export default function ResizablePanes({ left, right }: ResizablePanesProps) {
  // We use a responsive layout: on mobile it stacks vertically, on desktop it splits side-by-side.
  // react-resizable-panels calculates exact flex widths under the hood using Javascript
  // and updates the flex-basis CSS dynamically as you drag the resize handle.
  return (
    <div className="h-screen w-full flex flex-col md:flex-row bg-white">
      {/* Mobile: Vertical layout */}
      <div className="md:hidden flex flex-col h-full w-full overflow-hidden">
        <div className="flex-1 min-h-[40vh] border-b border-gray-200 relative overflow-hidden">
          {left}
        </div>
        <div className="flex-1 relative overflow-hidden">
          {right}
        </div>
      </div>

      {/* Desktop: Horizontal resizable layout */}
      <div className="hidden md:flex h-full w-full">
        <PanelGroup orientation="horizontal">
          <Panel defaultSize={55} minSize={25} className="h-full">
            {left}
          </Panel>
          <PanelResizeHandle className="w-1 bg-gray-200 hover:bg-gray-300 transition-colors cursor-col-resize flex items-center justify-center group relative">
            <div className="h-8 w-1 rounded-full bg-gray-400 group-hover:bg-gray-500 absolute" />
          </PanelResizeHandle>
          <Panel defaultSize={45} minSize={25} className="h-full">
            {right}
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}
