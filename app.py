from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from ortools.sat.python import cp_model

app = FastAPI(title="Truck Load Optimizer API", version="1.2.0")

# -------------------------
# CORS (Cross-Origin) FIX
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Data models ----------
class OrderIn(BaseModel):
    order: str
    p: str
    v: int = Field(..., ge=0)  # requested volume (L)
    haz: bool
    maxfill: int = Field(..., ge=1, le=100)  # max fill % per compartment


class CompartmentIn(BaseModel):
    capacity: int = Field(..., ge=1)
    baffleplate: bool = False


# ---- NEW: progress snapshot (no conversions inside optimizer!) ----
class OrderProgress(BaseModel):
    """
    Between executions, the calling app sends remaining quantities for each order
    in BOTH units:
      - remaining_l: liters (for compartment feasibility / maxfill / sloshing)
      - remaining_kg: kg-in-air (for truck gross constraint)
    If an order should not be planned anymore, set both to 0.
    """
    order: str
    remaining_l: int = Field(..., ge=0)
    remaining_kg: float = Field(..., ge=0)


class WeighingEvent(BaseModel):
    """
    We always weigh BEFORE starting the order and AFTER finishing the order.
    Delta (after - before) = net weight loaded for that order (kg).
    """
    order: str
    weigh_before_kg: float = Field(..., ge=0)
    weigh_after_kg: float = Field(..., ge=0)

    @property
    def delta_net_loaded_kg(self) -> float:
        return self.weigh_after_kg - self.weigh_before_kg


class CompartmentExecution(BaseModel):
    """
    Actual execution line: an order execution loaded into a compartment.
    Same compartment can receive multiple executions over time.
    """
    seq: int = Field(..., ge=1, description="Execution sequence number (1..n)")
    compartment: int = Field(..., ge=1, description="1-based compartment index")
    order: str
    actual_volume_l: Optional[int] = Field(None, ge=0, description="Actual loaded volume for this execution (L)")
    note: Optional[str] = None


class ExecutionState(BaseModel):
    """
    State passed between executions.
    Optimizer does NOT do any L<->kg conversions.
    """
    allowed_max_gross_kg: Optional[float] = Field(None, ge=0)
    current_gross_kg: Optional[float] = Field(
        None, ge=0, description="Latest gross weigh (typically weigh-after of last executed order)"
    )

    # Remaining quantities for planning (authoritative inputs)
    order_progress: List[OrderProgress] = Field(default_factory=list)

    # Optional traceability / audit
    weighings: List[WeighingEvent] = Field(default_factory=list)
    executions: List[CompartmentExecution] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_order_progress(self):
        seen = set()
        for op in self.order_progress:
            if op.order in seen:
                raise ValueError(f"Duplicate order_progress entry for order '{op.order}'.")
            seen.add(op.order)
        return self


class OptimizeRequest(BaseModel):
    orders: List[OrderIn]
    compartments: List[CompartmentIn]
    state: Optional[ExecutionState] = None


class LoadLine(BaseModel):
    order: str
    product: str
    volume_l: int
    weight_kg: float
    compartment: int
    hazardous: bool
    baffle: bool
    capacity_l: int
    fill_percent: float


class StateEcho(BaseModel):
    allowed_max_gross_kg: Optional[float] = None
    current_gross_kg: Optional[float] = None
    order_progress: List[OrderProgress] = Field(default_factory=list)
    weighings: List[WeighingEvent] = Field(default_factory=list)
    executions: List[CompartmentExecution] = Field(default_factory=list)


class OptimizeResponse(BaseModel):
    total_volume_loaded_l: int
    total_weight_loaded_kg: float
    loadplan: List[LoadLine]
    status: str
    state: Optional[StateEcho] = None


# ---------- Helpers ----------
def _state_echo(state: Optional[ExecutionState]) -> Optional[StateEcho]:
    if not state:
        return None
    return StateEcho(
        allowed_max_gross_kg=state.allowed_max_gross_kg,
        current_gross_kg=state.current_gross_kg,
        order_progress=list(state.order_progress or []),
        weighings=list(state.weighings or []),
        executions=list(state.executions or []),
    )


def _progress_map(state: Optional[ExecutionState]) -> Dict[str, OrderProgress]:
    if not state or not state.order_progress:
        return {}
    return {op.order: op for op in state.order_progress}


# ---------- API ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    orders = req.orders
    compartments = req.compartments
    state = req.state
    echo = _state_echo(state)

    if not orders:
        raise HTTPException(status_code=400, detail="No orders provided.")
    if not compartments:
        raise HTTPException(status_code=400, detail="No compartments provided.")

    n_orders = len(orders)
    n_c = len(compartments)

    # Map order -> remaining (L, kg) if provided; else fallback to full request (L) and kg=0
    # (kg=0 fallback means optimizer can still produce a volume plan; gross constraint won't bind.)
    pmap = _progress_map(state)

    remaining_l = []
    remaining_kg = []
    for o in range(n_orders):
        oid = orders[o].order
        if oid in pmap:
            remaining_l.append(int(pmap[oid].remaining_l))
            remaining_kg.append(float(pmap[oid].remaining_kg))
        else:
            remaining_l.append(int(orders[o].v))
            remaining_kg.append(0.0)

    # Build CP-SAT model
    model = cp_model.CpModel()

    v_l = {}      # liters assigned to (o,c)
    w_kg = {}     # kg assigned to (o,c)
    x = {}        # assignment (o,c) - still no blending in a compartment

    # --- Variables ---
    for o in range(n_orders):
        for c in range(n_c):
            cap_l = compartments[c].capacity

            # Max liters per compartment for this order (maxfill rule)
            max_fill_l = int(cap_l * orders[o].maxfill / 100)
            max_possible_l = min(cap_l, remaining_l[o], max_fill_l)

            # liters variable
            v_l[o, c] = model.NewIntVar(0, max_possible_l, f"vL_{o}_{c}")
            x[o, c] = model.NewBoolVar(f"x_{o}_{c}")

            model.Add(v_l[o, c] > 0).OnlyEnforceIf(x[o, c])
            model.Add(v_l[o, c] == 0).OnlyEnforceIf(x[o, c].Not())

            # kg variable: since no conversion is allowed, we model kg separately.
            # Upper bound is remaining_kg[o] (rounded up as int grams?).
            # CP-SAT is integer; use grams to keep precision.
            max_possible_g = int(round(remaining_kg[o] * 1000))
            w_kg[o, c] = model.NewIntVar(0, max_possible_g, f"wG_{o}_{c}")

            # Link weight usage to assignment too:
            model.Add(w_kg[o, c] > 0).OnlyEnforceIf(x[o, c])
            model.Add(w_kg[o, c] == 0).OnlyEnforceIf(x[o, c].Not())

    # --- Constraints per compartment ---
    for c in range(n_c):
        cap_l = compartments[c].capacity

        # liters capacity
        model.Add(sum(v_l[o, c] for o in range(n_orders)) <= cap_l)

        # no blending: at most one order per compartment
        model.Add(sum(x[o, c] for o in range(n_orders)) <= 1)

        # hazardous redundancy kept
        model.Add(sum(x[o, c] for o in range(n_orders) if orders[o].haz) <= 1)

    # --- Constraints per order ---
    for o in range(n_orders):
        # liters remaining
        model.Add(sum(v_l[o, c] for c in range(n_c)) <= remaining_l[o])

        # kg remaining (grams)
        max_g = int(round(remaining_kg[o] * 1000))
        model.Add(sum(w_kg[o, c] for c in range(n_c)) <= max_g)

    # --- Sloshing (liters-based) ---
    for o in range(n_orders):
        if not orders[o].haz:
            continue
        for c in range(n_c):
            cap_l = compartments[c].capacity
            if cap_l > 7500 and not compartments[c].baffleplate:
                low = int(cap_l * 0.2)
                high = int(cap_l * 0.8)

                is_low = model.NewBoolVar(f"is_low_{o}_{c}")
                is_high = model.NewBoolVar(f"is_high_{o}_{c}")

                model.Add(v_l[o, c] <= low).OnlyEnforceIf(is_low)
                model.Add(v_l[o, c] >= high).OnlyEnforceIf(is_high)
                model.AddBoolOr([is_low, is_high]).OnlyEnforceIf(x[o, c])

    # --- Truck gross constraint (kg, grams) ---
    if state and state.allowed_max_gross_kg is not None and state.current_gross_kg is not None:
        allowed_g = int(round(state.allowed_max_gross_kg * 1000))
        current_g = int(round(state.current_gross_kg * 1000))
        # current + planned <= allowed
        model.Add(current_g + sum(w_kg[o, c] for o in range(n_orders) for c in range(n_c)) <= allowed_g)

    # --- Objective ---
    # Since liters and kg are independent (no conversion), choose what you want to maximize.
    # Here: maximize liters first, then kg as tie-breaker.
    total_l = model.NewIntVar(0, sum(remaining_l), "total_l")
    model.Add(total_l == sum(v_l[o, c] for o in range(n_orders) for c in range(n_c)))

    total_g = model.NewIntVar(0, int(round(sum(remaining_kg) * 1000)), "total_g")
    model.Add(total_g == sum(w_kg[o, c] for o in range(n_orders) for c in range(n_c)))

    model.Maximize(total_l * 1_000_000 + total_g)

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return OptimizeResponse(
            total_volume_loaded_l=0,
            total_weight_loaded_kg=0.0,
            loadplan=[],
            status="NO_FEASIBLE_SOLUTION",
            state=echo,
        )

    # --- Build response ---
    loadplan: List[LoadLine] = []
    total_loaded_l = 0
    total_loaded_g = 0

    for o in range(n_orders):
        for c in range(n_c):
            vol_l = solver.Value(v_l[o, c])
            w_g = solver.Value(w_kg[o, c])
            if vol_l > 0 or w_g > 0:
                cap_l = compartments[c].capacity
                total_loaded_l += vol_l
                total_loaded_g += w_g

                loadplan.append(
                    LoadLine(
                        order=orders[o].order,
                        product=orders[o].p,
                        volume_l=int(vol_l),
                        weight_kg=round(w_g / 1000.0, 3),
                        compartment=c + 1,
                        hazardous=orders[o].haz,
                        baffle=compartments[c].baffleplate,
                        capacity_l=cap_l,
                        fill_percent=round((vol_l / cap_l) * 100.0, 1) if cap_l > 0 else 0.0,
                    )
                )

    loadplan.sort(key=lambda r: (r.order, r.compartment))

    return OptimizeResponse(
        total_volume_loaded_l=int(total_loaded_l),
        total_weight_loaded_kg=round(total_loaded_g / 1000.0, 3),
        loadplan=loadplan,
        status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        state=echo,
    )
