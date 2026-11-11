# Draw packets and command lists (stages 1 and 2 of the GPUBack command list)

GPUBack prepares every graphics draw in two stages. Stage A reads what the draw needs from the
Maxwell registers, dirty flags and guest memory as they are at the draw and writes it into a
`DrawPacket` (`src/video_core/renderer_vulkan/vk_draw_packet.h`). Stage B records the draw from
that packet. In stage 1 both run back to back on GPUBack in `RasterizerVulkan::PrepareDraw`;
there is no second thread and no switch back to the old path. Later stages move B behind a queue,
so B must not depend on the registers or dirty flags.

## What a packet holds

| Part | Contents |
|---|---|
| Header | Sequence number (one per prepared draw, for stage 4's ordering), pipeline, framebuffer, `DrawParams` or a copy of the indirect parameters, mesh and mesh emulation counts |
| Buffers (`VideoCommon::GraphicsBufferPacket`) | Index binding (when `Dirty::IndexBuffer` was set), index format, first, count and inline indexes; vertex streams to resolve or bind again with their strides; transform feedback bindings; storage and texture buffer bindings per stage, read from the constant buffers; indirect and count buffers; whether A uploaded buffer data |
| Descriptors | Where each stage's storage and texel buffer entries go in the descriptor queue's payload (A writes the other entries there itself); the rescaling and render area push constants |
| Flags | Transform feedback (ends the render pass and counts streamed bytes), sampled render target under a pending fragment barrier, rescaling, the zpass counter enable |
| Dynamic state | A presence mask and the values of each group whose dirty flag A took |

The dynamic state is a delta, not a register snapshot: a group is present only when A's
`Touch*` flag (or the state tracker's check) fired. `ReadDynamicGroup` in `vk_rasterizer.cpp`
computes each group's values; A and the verify both call it.

`DrawPacket::PayloadBytes` counts the bytes A wrote for the draw: the header fields the draw
uses (no indirect parameters for a direct draw, no mesh, index, transform feedback or indirect
fields where the draw has none), its lists by their used entries, and the dynamic state groups
present. Descriptor entries go to the descriptor queue, not the packet. EDEN_NX2_DRAW_PACKET_VERIFY
logs the average.

## Stage A

- Pipeline lookup (or the fast draw's reusable pipeline), texture resolve, texture memo, the
  render targets and the feedback loop check.
- Storage buffer addresses read from the constant buffers, texture buffer addresses from the
  resolved image views.
- `BufferCache::ReadGraphicsGeometry` reads the index, vertex, transform feedback and indirect
  buffers and takes `Dirty::IndexBuffer`, `Dirty::VertexBuffers` and `Dirty::VertexBuffer0+i`.
- The uniform buffers are resolved (`UpdateGraphicsUniformBuffers`, with its own retry loop) and
  bound: the stream buffer copies happen here, so they copy guest memory as it is at the draw.
- A acquires the descriptor payload and writes the entries in place. Per stage: the uniform buffer
  entries, room for the storage and texel buffer entries (`UpdateDescriptorQueue::Skip`, its range
  in the packet), the image entries (a delta draw copies its pipeline's kept image entries). Then
  the mesh emulation entries.
- The dynamic state flags are taken and their groups read into the packet. When binding the
  pipeline in B will make the state enables dirty again (`Scheduler::RefreshesStateEnable`, after
  a pipeline without extended dynamic state), A takes them too, so they are recorded with this
  draw as before the split.

## Stage B

`GraphicsPipeline::Record` and the rest of `PrepareDraw` use only the packet and state the caches
own:

1. Request the render pass end for transform feedback.
2. Resolve every buffer (retry loop): A's index and vertex bindings go to the buffer cache first;
   the storage, texture, transform feedback and indirect bindings are bound and resolved from the
   packet in the same pass. Then synchronize and upload, and bind the index, vertex, transform
   feedback and indirect buffers.
3. Per stage, write the storage and texel buffer entries (bound now) into the room A left
   (`BeginFill`/`EndFill`). Another count than A left fails the draw in `ConfigureDraw`, as a
   short payload did before.
4. The post-copy barrier, the fragment barrier check, the render pass, pipeline, descriptor set
   and push constants (`ConfigureDraw`).
5. The dynamic state records, the query counter enables and the draw command.

If B can't record the draw, it still records the dynamic state, so the flags A took are not lost.
If a submit happened since A (the scheduler's tick moved), the new command buffer has no dynamic
state, so B takes every group again before it records them.

## Owners and joins (stage 3)

`VideoCommon::DrawStages` (`src/video_core/nx2_draw_stages.h`) marks which stage the thread that
draws is in. `PrepareDraw` enters A after `BeginCommandList`, B at the splice, and leaves both
at the draw's end. Between draws (methods, clears, compute) either stage's state may be used.
With `EDEN_NX2_SPLIT_CHECK=1`, each owner's entry points check the marker and log the first 16
violations. In stage 4 the same checks become thread checks.

| Owner | State | Checked entry points |
|---|---|---|
| A | Registers and dirty flags (the state tracker's `Touch*` take them) | `UpdateDynamicStates`, `ReadGraphicsGeometry` |
| A | Texture cache calls | Not checked inside the texture cache (EGLA-98's files). No B path calls it, apart from the diagnostics listed below |
| A | Uniform resolve, binds and the uniform stream; storage and texture buffer reads into the packet | `UpdateGraphicsUniformBuffers`, `BindHostGraphicsUniformBuffers`, `BindGraphicsStorageBuffer`, `BindGraphicsTextureBuffer` |
| A | Descriptor payload acquire and writes | `GraphicsPipeline::Prepare`/`PrepareDelta`, `AcquireDescriptors`, `UpdateDescriptorQueue::Skip` |
| B | The scheduler outside A's command list | `Scheduler::Record` when not capturing, `RequestRenderpass` |
| B | Buffer cache binding, resolve, sync and upload | `UpdateGraphicsBuffers`, `BindHostGeometryBuffers`, `BindHostStageBuffers` |
| B | The descriptor room A left | `UpdateDescriptorQueue::BeginFill` |
| B | Dynamic state records | `GraphicsPipeline::Record`, `RecordDynamicStates` |

Join points are where stage 4's A waits until B consumed every packet A sent. On one thread a
join only counts. Counting runs with `EDEN_NX2_GPU_PROFILE` or `EDEN_NX2_SPLIT_CHECK`; otherwise
each site tests one flag. The back's GPU profile line names each site in its "draw stage joins
per second" part.

- **Front waits:** every drain or position wait of the split's front that waited, by drain site
  (acquire, list writers, list sync, list flush, fence release and the others, ring full
  excepted). In stage 4 the back's consumed position has to include B.
- **Writes over packets in flight:** DMA launches, I2M and inline writes, CB_DATA writes and
  query writes are checked against the device ranges B read for the last 64 packets. This is the
  stage 0 probe's ring, standing in for stage 4's packet ring. They count as "joins of writes
  checked".
- **DMA image<->buffer copies:** the texture cache calls the buffer cache.
- **Guest flush reads:** `GPU::OnCPURead` flushes that aren't already in flight, and buffer cache
  `FlushRegion` calls.
- **Present.**
- **Stage A on B's state:**
  - A submit or dispatch during A (`SubmitExecution`, `DispatchWork`).
  - A full command list.
  - A render pass end during A. The texture cache ends the scheduler's render pass for uploads,
    copies and clears.
  - A clear deferred or realized during A.
  - A reading scheduler state: `HasFragmentBarrier`, `RefreshesStateEnable`,
    `IsRenderPassActive` and `RenderPassSerial`, which the fast draw's reuse checks call.
  - A creating or deleting a buffer, and A uploading buffer data. Both come from the uniform
    resolve, which shares `FindBuffer` and `SynchronizeBuffer` with B.
- **B on A's state:** B takes the dynamic state from the registers again after a submit since
  A.

While a join holds, the owner checks treat the thread as between draws, as stage 4's B would be
idle.

### Still shared, not yet a join

- The guest descriptor queue: A acquires and writes the payload, and B fills the room A left. In
  stage 4, A must not acquire payload B hasn't recorded yet, and `TickFrame` must not reuse it
  before B has.
- The command list is the scheduler's, not the packet's. Stage 4 needs one per packet in the
  ring, and the scheduler state A changes has to travel with it. The A-on-B-state join counts
  show how often that happens.
- B reads the page table for multi-range storage segments, and guest memory for uploads and uint8
  index widening.
- A writes the vertex repack layout into the buffer cache runtime and the mesh emulation buffers
  into the pipeline. B's `ConfigureDraw` reads the mesh group count and task buffers from the
  pipeline.
- The diagnostics between B's configure and its dynamic state (render state trace, frame dump,
  Ocarina tracking, indirect trace) read the registers and caches directly.

## Ordering changes against the old path

- A's recordings (texture uploads, uniform uploads) now come before B's buffer uploads.
- A resolves the uniform buffers before B resolves the others. If B's resolve merges a uniform
  buffer away, this draw's descriptor still names the old buffer; delayed destruction keeps it
  alive and its contents are those of the draw.
- `DeleteBuffer` and `RefreshRepackedVertexBuffers` no longer set the index and vertex dirty
  flags. They set resolve and rebind bits the buffer cache owns, so a deletion during a draw no
  longer records the vertex input again.

## Command lists (stage 2)

Stage A records into a command list, not the command buffer. `RasterizerVulkan::PrepareDraw`
calls `Scheduler::BeginCommandList` after `FlushWork`. From then on, `Scheduler::Record` and
`RecordWithUploadBuffer` put commands in the scheduler's one reusable list (a `CommandChunk`).
That covers the texture cache runtime's uploads, copies, clears and render target barriers, the
uniform buffer copies, render pass ends, and anything else A records. The texture cache itself is
unchanged.

`Scheduler::SpliceCommandList` runs right before `GraphicsPipeline::Record`, or at any early
return from the draw. It moves the list's commands, in order, to the end of the chunk. B's
commands follow them.

- **Empty lists:** splicing an empty list costs one flag store and one test. No list is
  allocated per draw.
- **Joins:** a submit (`Flush`, `Finish`) or a dispatch (`DispatchWork`, `WaitWorker`)
  during stage A splices the list first, and records its own commands into the chunk. A full
  list is spliced early too. Capturing continues after either.
- **State:** the scheduler's state (render pass, pipeline, deferred clears) still changes when A
  calls into it. On one thread this matches the order the commands land in, because nothing
  records between A and the splice. Where A changes it, stage 3 has a join point (below).

`EDEN_NX2_CMDLIST_VERIFY=1` gives every recorded command a type tag and a hash of its captured
arguments. It numbers commands in the order they are recorded, which is the order the old direct
path put them in the command buffer, and checks that the command buffer receives them in that
order. A spliced command that comes out of the move as a different type also counts as a
difference. The first 16 differences get their own line. Every 10 s a `Command list verify`
line logs:
- commands checked and those that differ (must stay 0);
- draws, and how many of them had a list, with the commands those lists held;
- joins by a submit or dispatch, and by a full list.

## Stage B on its own thread (stage 4, test only)

`EDEN_NX2_BACK_STAGES=1` runs stage B on a thread of its own (`GPUStageB`) while the split's back
draws. It is off by default; off, the draw runs both stages inline as in stage 3. The rasterizer
starts the ring at the back's first draw and stops it as that thread exits. Accurate GPU fences
and legacy Aggressive VRAM mode keep draws inline; they need submitting waits during buffer resolve.

- Stage A fills a packet slot and records everything else into the slot's command list
  (`Scheduler::PacketList`): uploads, copies, clears, barriers. Render pass ends, deferred clears,
  the scheduler's part of an invalidate, deferred fragment barriers and dispatches go into the
  list as splice ops, which B applies in place as it splices.
- A hands packets over in batches of 32 (`BackRing`, 64 slots), or at once at a join. B spins
  3 µs, then sleeps on a futex. A blocks when the ring is full.
- B records each packet holding the buffer cache's lock: it splices the list, sets the vertex
  repack layout from the packet, binds buffers and descriptors, records the dynamic state, and
  records the draw. B keeps a running copy of the dynamic state and records all of it after a
  submit of its own.
- A keeps the state tracker. Fast draws read A's view of the scheduler (`BackView`): the
  pipeline and framebuffer of the packets A handed over, until A changes the scheduler state.
- Uniform buffers: A streams the small ones at the draw. The rest go to B in `uniform_binds`,
  with their device address. The uniform keep optimization is disabled while B runs separately.
- Joins wait until B has consumed every packet, then splice A's open list:
  - the buffer cache's lock, and any direct scheduler use: submits, waits, render pass requests
    and deferred clears;
  - clears, compute, queries and conditional rendering, fences, flushes, frames, copies,
    channels, barriers and the idle hook;
  - a descriptor payload wrap;
  - writes over a range a packet in flight reads (A notes those ranges at each packet);
  - draws a diagnostic reads (render trace, frame dump, fast draw verify, pending write trace,
    indirect read-back, native capture, Ocarina DLSS tracking). Those draws run both stages on A.
- The split counts the back's position for the front's waits only once B has recorded the
  packets made before it. A drain hands the open batch over at once.
- `EDEN_NX2_SPLIT_CHECK` accepts scheduler use from B's thread.
- Lifetimes of what packets point at:
  - Pipeline cache GC joins B before it evicts a pipeline. Evicted pipelines are destroyed once
    their tick is free.
  - The texture cache joins B before it creates a framebuffer: a new one can move the others in
    the slot vector. It also joins before it removes framebuffers. B's parked deferred clear lands
    directly after that join. A foreign-thread unmap removes framebuffers from the lookup table
    but retains their objects until the drawing thread can join and retire them.
  - The texture cache destroys images and views on the frame ring, after a TickFrame, which joins
    first.
- B must not submit the current scheduler tick. A tags its staging memory and arenas with that
  tick, including packets B has not recorded yet. A reserves descriptor capacity before preparing
  each draw, and joins and finishes before it needs to rewind the descriptor frame. Reused-frame
  waits also run on A, before packet preparation. The B-side exhaustion branch skips and counts
  the draw rather than submitting; reaching it is a failed capacity-plan invariant. Multi-range
  retirement can grow its retained-resource list on B instead of forcing a submitting wait.
- B keeps a complete dynamic-state copy as a diagnostic backstop. The `Back stages` line counts
  any unexpected B submit; this count and skipped-descriptor-draw count must stay zero.
- Emulated mesh draws carry their dispatch (task buffers, group count, arena words) in the
  packet. A sets the pipeline's copy for the next draw while B records an earlier one.
- Transform-feedback query callbacks use the packet's enable, output topology, patch count and
  buffer stream/stride state, including callbacks triggered while splicing render-pass ends.
- A draw B cannot record still sets the native macro engine's failure flag. A receives it at its
  next draw, join or ring stop.
- The ring starts on the split's back and stops there as the back exits. The stop clears that
  thread's capture and the ring's hooks before anything else records. An epoch also invalidates
  captures left on other threads, so a later ring cannot revive an old capture pointer.

`EDEN_NX2_BACK_STAGES_VERIFY=1` makes A hash each range B reads (its first 256 KiB). B hashes
the range again before it records the packet and checks that packets arrive in order. B logs the
first 16 differences. Every 10 s a `Back stages verify` line gives the packets, the ranges
checked, those that changed after A built the packet, the packets out of order, and packets
recorded after the address-space serial changed. B does not resolve the addresses again: the
packet holds device addresses only. The content hash and remap count cover the first PC run;
they do not prove that a GPU address still translates to the same device address. A
`Back stages` line gives the ring's hand-overs, joins, full-ring waits, A's wait time, and B's
sleeps and busy time. With `EDEN_NX2_GPU_PROFILE`, the profile line counts the joins by site.

## Verify

`EDEN_NX2_DRAW_PACKET_VERIFY=1` resolves each draw again from the live registers between A and B
and compares: draw or indirect parameters, the index binding, vertex bindings and strides,
storage buffer bindings, transform feedback and zpass enables, render area, framebuffer,
rescaling, and every dynamic state group the packet carries. It logs the first 16 differences and
every 10 s the draws checked, how many differed and the average packet size. Uniform and image
descriptors are not resolved again (that would upload uniforms and prepare images a second time);
the fast draw, texture memo and delta descriptor verifies cover them. In stage 1 every count must
stay 0, since nothing runs between A and B.
