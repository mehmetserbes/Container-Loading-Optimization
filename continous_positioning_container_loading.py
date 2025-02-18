import pulp
import matplotlib.pyplot as plt

def safe_val(var):
    # pulp.value(var) None dönebilir, None durumunda 0 döndürmek için
    val = pulp.value(var)
    return val if val is not None else 0

#############################################
# 1. VERİ GİRİŞİ - Araçlar ve Boru Tipleri Tanımlama
#############################################
vehicles = {
    'V1': {'L': 7000, 'W': 2500, 'H': 2800}  # Örnek: 7000x2500x2800 mm
}

pipe_types = {
    'P1': {'q': 10,  'd': 250, 'I': 240, 'm': 250, 'T': 40},
    'P2': {'q': 15,  'd': 110, 'I': 100, 'm': 125, 'T': 30},
    'P3': {'q': 20,  'd': 63,  'I': 60,  'm': 63,  'T': 0}
}

L_pipe = 5800  # Boruların Y yönündeki kapladığı uzunluk

# Her boru tipi için efektif yükseklik hesaplanıyor
for p in pipe_types:
    if pipe_types[p]['T'] > 0:
        pipe_types[p]['h_eff'] = pipe_types[p]['m'] + pipe_types[p]['T']
    else:
        pipe_types[p]['h_eff'] = 0.85 * pipe_types[p]['m']

# Her boru siparişi için benzersiz ID oluşturuluyor
J = []  # Tüm boru ID'lerinin listesi
pipe_data = {}
for p in pipe_types:
    for i in range(pipe_types[p]['q']):
        jid = f"{p}_{i+1}"
        J.append(jid)
        pipe_data[jid] = {
            'type': p,
            'd_eff': pipe_types[p]['m'],    # Etkili dış çap
            'h': pipe_types[p]['h_eff'],      # Etkili yükseklik
            'I': pipe_types[p]['I'],          # İç çap
            'm': pipe_types[p]['m']           # Dış çap
        }

# Aynı tip boruların nested olmaması için tanımlanan kısıt
same_type = {}
for j in J:
    for i in J:
        if j != i:
            same_type[(j, i)] = 1 if pipe_data[j]['type'] == pipe_data[i]['type'] else 0

# Nested yerleştirme uygunluğunu kontrol eden parametreler (epsilon kaldırıldı)
can_host2 = {}
can_host3 = {}
for j in J:
    for i in J:
        if j != i:
            if 1.05 * pipe_data[j]['m'] < pipe_data[i]['I']:
                can_host2[(j, i)] = 1
                can_host3[(j, i)] = 1
            else:
                can_host2[(j, i)] = 0
                can_host3[(j, i)] = 0

V_list = list(vehicles.keys())

#############################################
# 2. MODELİN KURULMASI (Optimizasyon Modeli)
#############################################
model = pulp.LpProblem("3D_Nested_Stacking", pulp.LpMinimize)
M_big = 1e7  # Big-M değeri

# x[j,v]: Boru j'nin konteyner v'ye atanması (binary)
x = pulp.LpVariable.dicts("x", [(j, v) for j in J for v in V_list], 0, 1, pulp.LpBinary)
for j in J:
    model += pulp.lpSum(x[(j, v)] for v in V_list) == 1

# l1, l2, l3: Borunun hangi seviyede olduğunu gösterir (outer, nested level-2, nested level-3)
l1 = pulp.LpVariable.dicts("l1", J, 0, 1, pulp.LpBinary)
l2 = pulp.LpVariable.dicts("l2", J, 0, 1, pulp.LpBinary)
l3 = pulp.LpVariable.dicts("l3", J, 0, 1, pulp.LpBinary)
for j in J:
    model += l1[j] + l2[j] + l3[j] == 1

# Nested atama değişkenleri
h2 = pulp.LpVariable.dicts("h2", [(j, i, v) for j in J for i in J if j != i for v in V_list], 0, 1, pulp.LpBinary)
h3 = pulp.LpVariable.dicts("h3", [(j, i, v) for j in J for i in J if j != i for v in V_list], 0, 1, pulp.LpBinary)

# Aynı tip boruların nested olmaması
for v in V_list:
    for j in J:
        for i in J:
            if j != i and same_type[(j, i)] == 1:
                model += h2[(j, i, v)] == 0
                model += h3[(j, i, v)] == 0

# Fiziksel nested kısıtları (nested atama yalnızca uygun hostlarda yapılabilir)
for v in V_list:
    for j in J:
        for i in J:
            if j != i:
                model += h2[(j, i, v)] <= can_host2[(j, i)]
                model += h3[(j, i, v)] <= can_host3[(j, i)]

# Her host boruda en fazla 1 nested boru yer alabilir.
for v in V_list:
    for i in J:
        model += pulp.lpSum(h2[(j, i, v)] for j in J if j != i) <= 1
        model += pulp.lpSum(h3[(j, i, v)] for j in J if j != i) <= 1

# Host borunun seviyesine uygun nested yerleştirme:
for v in V_list:
    for j in J:
        for i in J:
            if j != i:
                model += h2[(j, i, v)] <= l1[i]
                model += h3[(j, i, v)] <= l2[i]

# z1, z2, z3: x[j,v] ile l1, l2, l3'nin çarpımını modelleyen yardımcı değişkenler
z1 = pulp.LpVariable.dicts("z1", [(j, v) for j in J for v in V_list], 0, 1, pulp.LpBinary)
z2 = pulp.LpVariable.dicts("z2", [(j, v) for j in J for v in V_list], 0, 1, pulp.LpBinary)
z3 = pulp.LpVariable.dicts("z3", [(j, v) for j in J for v in V_list], 0, 1, pulp.LpBinary)

for j in J:
    for v in V_list:
        model += z1[(j, v)] <= x[(j, v)]
        model += z1[(j, v)] <= l1[j]
        model += z1[(j, v)] >= x[(j, v)] + l1[j] - 1

        model += z2[(j, v)] <= x[(j, v)]
        model += z2[(j, v)] <= l2[j]
        model += z2[(j, v)] >= x[(j, v)] + l2[j] - 1

        model += z3[(j, v)] <= x[(j, v)]
        model += z3[(j, v)] <= l3[j]
        model += z3[(j, v)] >= x[(j, v)] + l3[j] - 1

# Nested atama kısıtları:
for j in J:
    for v in V_list:
        model += pulp.lpSum(h2[(j, i, v)] for i in J if i != j) == z2[(j, v)]
        model += pulp.lpSum(h3[(j, i, v)] for i in J if i != j) == z3[(j, v)]

# Konteyner kullanım karar değişkeni
y = pulp.LpVariable.dicts("y", V_list, 0, 1, pulp.LpBinary)
for j in J:
    for v in V_list:
        model += x[(j, v)] <= y[v]
for v in V_list:
    model += pulp.lpSum(z1[(j, v)] for j in J) >= y[v]

# Outer boruların 3D pozisyonlarını belirleyen değişkenler (X, Y, Z koordinatları)
X_pos = pulp.LpVariable.dicts("X", [(j, v) for j in J for v in V_list], 0, None, pulp.LpContinuous)
Y_pos = pulp.LpVariable.dicts("Y", [(j, v) for j in J for v in V_list], 0, None, pulp.LpContinuous)
Z_pos = pulp.LpVariable.dicts("Z", [(j, v) for j in J for v in V_list], 0, None, pulp.LpContinuous)

# Outer boruların konteyner sınırları içerisinde kalması için kısıtlar
for v in V_list:
    W_v = vehicles[v]['W']
    L_v = vehicles[v]['L']
    H_v = vehicles[v]['H']
    for j in J:
        model += X_pos[(j, v)] + pipe_data[j]['d_eff'] <= W_v + M_big*(1 - z1[(j, v)])
        model += Y_pos[(j, v)] + L_pipe <= L_v + M_big*(1 - z1[(j, v)])
        model += Z_pos[(j, v)] + pipe_data[j]['h'] <= H_v + M_big*(1 - z1[(j, v)])

#############################################
# 3. STACKING (Yığma) DEĞİŞKENLERİ
#############################################
# Outer boruların yerleşim şekli: taban (floor) veya stacking (başka borunun üzerinde)
floor_ = pulp.LpVariable.dicts("floor", [(j, v) for j in J for v in V_list], 0, 1, pulp.LpBinary)
on_top = pulp.LpVariable.dicts("on_top", [(j, i, v) for j in J for i in J if j != i for v in V_list], 0, 1, pulp.LpBinary)

for j in J:
    for v in V_list:
        model += floor_[(j, v)] + pulp.lpSum(on_top[(j, i, v)] for i in J if i != j) == z1[(j, v)]

for v in V_list:
    for j in J:
        # Eğer boru tabanda ise, Z pozisyonu 0 olmalıdır (big-M ile modellenir)
        model += Z_pos[(j, v)] <= M_big*(1 - floor_[(j, v)])
        model += Z_pos[(j, v)] >= -M_big*(1 - floor_[(j, v)])
        
for v in V_list:
    for j in J:
        for i in J:
            if i != j:
                # Stacking durumunda dikey ilişki: boru j, boru i’nin yüksekliği kadar üstte yer alır
                model += (Z_pos[(j, v)] - Z_pos[(i, v)] - pipe_data[i]['h']
                          <= M_big*(1 - on_top[(j, i, v)]))
                model += (Z_pos[(j, v)] - Z_pos[(i, v)] - pipe_data[i]['h']
                          >= -M_big*(1 - on_top[(j, i, v)]))
                
for v in V_list:
    for j in J:
        for i in J:
            if i != j:
                # Stacking durumunda yatay (X ve Y) koordinatlar eşitlenir
                model += (X_pos[(j, v)] - X_pos[(i, v)]
                          <= M_big*(1 - on_top[(j, i, v)]))
                model += (X_pos[(j, v)] - X_pos[(i, v)]
                          >= -M_big*(1 - on_top[(j, i, v)]))
                model += (Y_pos[(j, v)] - Y_pos[(i, v)]
                          <= M_big*(1 - on_top[(j, i, v)]))
                model += (Y_pos[(j, v)] - Y_pos[(i, v)]
                          >= -M_big*(1 - on_top[(j, i, v)]))

#############################################
# 4. 2D NON-OVERLAP KISITLARI (Tüm Outer Borular için: taban ve stacking)
#############################################
# outerPair: Outer (l1) olarak atanmış boru çiftlerinin aynı konteynerde birlikte yer alıp almadığını gösterir
outerPair = pulp.LpVariable.dicts("outerPair", [(i, j, v) for i in J for j in J for v in V_list if i < j], 0, 1, pulp.LpBinary)
for v in V_list:
    for i in J:
        for j in J:
            if i < j:
                model += outerPair[(i, j, v)] <= z1[(i, v)]
                model += outerPair[(i, j, v)] <= z1[(j, v)]
                model += outerPair[(i, j, v)] >= z1[(i, v)] + z1[(j, v)] - 1

# Delta (δ) değişkenleri: İki boru arasındaki yatay ayrımı belirler
delta_left = pulp.LpVariable.dicts("delta_left", [(i, j, v) for i in J for j in J if i < j for v in V_list], 0, 1, pulp.LpBinary)
delta_right = pulp.LpVariable.dicts("delta_right", [(i, j, v) for i in J for j in J if i < j for v in V_list], 0, 1, pulp.LpBinary)
delta_front = pulp.LpVariable.dicts("delta_front", [(i, j, v) for i in J for j in J if i < j for v in V_list], 0, 1, pulp.LpBinary)
delta_back = pulp.LpVariable.dicts("delta_back", [(i, j, v) for i in J for j in J if i < j for v in V_list], 0, 1, pulp.LpBinary)

# Non-overlap kısıtları (X-Y düzleminde, en az bir yön ayrımı sağlanmalı)
for v in V_list:
    for i in J:
        for j in J:
            if i < j:
                model += (X_pos[(i, v)] + pipe_data[i]['d_eff'] 
                          <= X_pos[(j, v)] + M_big*(1 - delta_left[(i, j, v)]) 
                             + M_big*(1 - outerPair[(i, j, v)]))
                model += (X_pos[(j, v)] + pipe_data[j]['d_eff'] 
                          <= X_pos[(i, v)] + M_big*(1 - delta_right[(i, j, v)]) 
                             + M_big*(1 - outerPair[(i, j, v)]))
                model += (Y_pos[(i, v)] + L_pipe 
                          <= Y_pos[(j, v)] + M_big*(1 - delta_front[(i, j, v)]) 
                             + M_big*(1 - outerPair[(i, j, v)]))
                model += (Y_pos[(j, v)] + L_pipe
                          <= Y_pos[(i, v)] + M_big*(1 - delta_back[(i, j, v)]) 
                             + M_big*(1 - outerPair[(i, j, v)]))
                model += (delta_left[(i, j, v)] + delta_right[(i, j, v)] +
                          delta_front[(i, j, v)] + delta_back[(i, j, v)]
                         ) >= outerPair[(i, j, v)]

#############################################
# 5. NESTING UYGUNLUK KISITLARI (Fiziksel uygunluk kontrolü)
#############################################
for v in V_list:
    for i in J:
        for j in J:
            if i != j:
                model += 1.05 * pipe_data[j]['m'] <= pipe_data[i]['I'] + M_big*(1 - h2[(j, i, v)])
                model += 1.05 * pipe_data[j]['m'] <= pipe_data[i]['I'] + M_big*(1 - h3[(j, i, v)])

#############################################
# 6. AMAÇ FONKSİYONU
#############################################
# Kullanılan konteyner sayısını minimize et
model += pulp.lpSum(y[v] for v in V_list)

#############################################
# 7. MODELİN ÇÖZÜLMESİ
#############################################
solver = pulp.COIN_CMD(msg=0)
model.solve(solver=solver)

print("Status:", pulp.LpStatus[model.status])
print("Objective:", pulp.value(model.objective))

# Çözüm bilgilerini yazdırma:
for v in V_list:
    if safe_val(y[v]) >= 0.5:
        print(f"\nContainer {v} used.")
        for j in J:
            if safe_val(x[(j, v)]) >= 0.5:
                if safe_val(l1[j]) >= 0.5:
                    Xv = safe_val(X_pos[(j, v)])
                    Yv = safe_val(Y_pos[(j, v)])
                    Zv = safe_val(Z_pos[(j, v)])
                    # Daire çizimi için yarıçap hesaplanıyor
                    r = pipe_data[j]['d_eff'] / 2.0
                    # Dairenin merkezi: (X + r, Z + r)
                    center = (Xv + r, Zv + r)
                    # Dairenin çizimi: silindirik görünüm
                    circle = plt.Circle(center, r, facecolor='lightblue', edgecolor='blue', alpha=0.8)
                    plt.gca().add_patch(circle)
                    plt.gca().text(center[0], center[1], j, ha='center', va='center', fontsize=8)
                elif safe_val(l2[j]) >= 0.5:
                    host = [i2 for i2 in J if i2 != j and safe_val(h2[(j, i2, v)]) > 0.5]
                    print(f"  {j} Nested L2 in {host}")
                else:
                    host = [i2 for i2 in J if i2 != j and safe_val(h3[(j, i2, v)]) > 0.5]
                    print(f"  {j} Nested L3 in {host}")

#############################################
# 8. GÖRSELLEŞTİRME
#############################################
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Container Vertical Cross-Section (Width vs. Height)")
ax.set_xlabel("Width (mm)")
ax.set_ylabel("Height (mm)")

for v in V_list:
    if safe_val(y[v]) < 0.5:
        continue
    W_v = vehicles[v]['W']
    H_v = vehicles[v]['H']
    # Konteyner sınırlarını çizelim
    cont_rect = plt.Rectangle((0, 0), W_v, H_v, fill=False, edgecolor='black', lw=2)
    ax.add_patch(cont_rect)
    ax.text(W_v*0.95, H_v*0.95, v, color='red', ha='right', va='top', fontsize=12)

    # Outer (l1 aktif) boruların pozisyonlarını çizelim
    for j in J:
        if safe_val(x[(j, v)]) >= 0.5 and safe_val(l1[j]) >= 0.5:
            Xv = safe_val(X_pos[(j, v)])
            Zv = safe_val(Z_pos[(j, v)])
            # Boru için çap: d_eff, yarıçap = d_eff/2
            r = pipe_data[j]['d_eff'] / 2.0
            # Dairenin merkezi (X+ r, Z+ r)
            center = (Xv + r, Zv + r)
            circle = plt.Circle(center, r, facecolor='lightblue', edgecolor='blue', alpha=0.8)
            ax.add_patch(circle)
            ax.text(center[0], center[1], j, ha='center', va='center', fontsize=8)

ax.set_xlim(0, vehicles['V1']['W'] + 100)
ax.set_ylim(0, vehicles['V1']['H'] + 100)
plt.tight_layout()
plt.show()



#Modelin potansiyel eksikleri:

#Yazdığım kodda sadece 3 level nesting (3 aşama iç içe koyma) mevcut, bu artırılabilir.

#Konteynır tiplerinin hepsi ayrı ayrı giriliyor. Belirlenen tip konteynır veya tırdan kaç adet mevcut olduğu da input olarak alınabilir.

#Altlı üstlü yerleştirilen borularda muf yönlerinin farklı olması kısıtı modelde mevcut değil.
