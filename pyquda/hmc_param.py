from typing import Literal

from .field import X, Y, Z, T
from .action.abstract import LoopParam, RationalParam


def gauge_loop_param(type: Literal["wilson", "symanzik_tree"], u_0: float):
    if type == "wilson":
        return LoopParam(
            path=[
                [X, Y, -X, -Y],
                [X, Z, -X, -Z],
                [X, T, -X, -T],
                [Y, Z, -Y, -Z],
                [Y, T, -Y, -T],
                [Z, T, -Z, -T],
            ],
            coeff=[
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        )
    elif type == "symanzik_tree":
        return LoopParam(
            path=[
                [X, Y, -X, -Y],
                [X, Z, -X, -Z],
                [X, T, -X, -T],
                [Y, Z, -Y, -Z],
                [Y, T, -Y, -T],
                [Z, T, -Z, -T],
                [X, X, Y, -X, -X, -Y],
                [X, X, Z, -X, -X, -Z],
                [X, X, T, -X, -X, -T],
                [Y, Y, X, -Y, -Y, -X],
                [Y, Y, Z, -Y, -Y, -Z],
                [Y, Y, T, -Y, -Y, -T],
                [Z, Z, X, -Z, -Z, -X],
                [Z, Z, Y, -Z, -Z, -Y],
                [Z, Z, T, -Z, -Z, -T],
                [T, T, X, -T, -T, -X],
                [T, T, Y, -T, -T, -Y],
                [T, T, Z, -T, -T, -Z],
            ],
            coeff=[
                1,
                1,
                1,
                1,
                1,
                1,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
                -1 / 20 / u_0**2,
            ],
        )


wilson_rational_param = {
    2: RationalParam(
        residue_molecular_dynamics=[
            1.0,
        ],
        offset_molecular_dynamics=[
            0.0,
        ],
        norm_pseudo_fermion=0.0,
        residue_pseudo_fermion=[
            1.0,
        ],
        offset_pseudo_fermion=[
            0.0,
        ],
    ),
    1: RationalParam(
        residue_molecular_dynamics=[
            0.00943108618345698,
            0.0122499930158508,
            0.0187308029056777,
            0.0308130330025528,
            0.0521206555919226,
            0.0890870585774984,
            0.153090120000215,
            0.26493803350899,
            0.466760251501358,
            0.866223656646014,
            1.8819154073627,
            6.96033769739192,
        ],
        offset_molecular_dynamics=[
            5.23045292201785e-05,
            0.000569214182255549,
            0.00226724207135389,
            0.00732861083302471,
            0.0222608882919378,
            0.0662886891030569,
            0.196319420401789,
            0.582378159903323,
            1.74664271771668,
            5.42569216297222,
            18.850085313508,
            99.6213166072174,
        ],
        norm_pseudo_fermion=6.10610118771501,
        residue_pseudo_fermion=[
            -5.90262826538435e-06,
            -2.63363387226834e-05,
            -8.62160355606352e-05,
            -0.000263984258286453,
            -0.000792810319715722,
            -0.00236581977385576,
            -0.00704746125114149,
            -0.0210131715847004,
            -0.0629242233443976,
            -0.190538104129215,
            -0.592816342814611,
            -1.96992441194278,
            -7.70705574740274,
            -46.55440910469,
            -1281.70053339288,
        ],
        offset_pseudo_fermion=[
            0.000109335909283339,
            0.000584211769074023,
            0.00181216713967916,
            0.00478464392272826,
            0.0119020708754186,
            0.0289155646996088,
            0.0695922442548162,
            0.166959610676697,
            0.400720136243831,
            0.965951931276981,
            2.35629923417205,
            5.92110728201649,
            16.0486180482883,
            53.7484938194392,
            402.99686403222,
        ],
    ),
}

staggered_rational_param = {
    ((0.5,), (1,)): RationalParam(
        norm_molecular_dynamics=1.3325011583706989e-01,
        residue_molecular_dynamics=[
            1.2669186056995732e-01,
            2.7378920166284243e-01,
            5.5644974578431738e-01,
            1.1865868336039220e00,
            2.7858758337966751e00,
            8.3158629779846862e00,
            5.3208853108759094e01,
        ],
        offset_molecular_dynamics=[
            1.0996250034766824e00,
            1.7746229220144820e00,
            3.7756006003660438e00,
            9.3414743860895957e00,
            2.5702192260694158e01,
            8.3642288108773357e01,
            4.6252616250061345e02,
        ],
        norm_pseudo_fermion=2.8919739467122496e00,
        residue_pseudo_fermion=[
            -2.3557845961480885e-02,
            -8.2885482286405326e-02,
            -2.1764644603751068e-01,
            -5.4479339852426578e-01,
            -1.3860680088812929e00,
            -3.7765935660639274e00,
            -1.2139134088997151e01,
            -5.8588994817562380e01,
            -1.0307790286845684e03,
        ],
        offset_pseudo_fermion=[
            1.1040557735606693e00,
            1.5736800437416782e00,
            2.7153853579224934e00,
            5.2775876670360722e00,
            1.1058396002583724e01,
            2.4733869901006777e01,
            6.0995784602031200e01,
            1.8772293711130985e02,
            1.2192272712884435e03,
        ],
        norm_fermion_action=3.4578458119819977e-01,
        residue_fermion_action=[
            3.5964179588188865e-02,
            8.7012555774795899e-02,
            1.7462541549477295e-01,
            3.4715574154454032e-01,
            7.0939528613321157e-01,
            1.5416834731507183e00,
            3.8299173393357995e00,
            1.3040678914547950e01,
            1.1265862119439132e02,
        ],
        offset_fermion_action=[
            1.0746987053604169e00,
            1.4873530879912891e00,
            2.5167732300465526e00,
            4.8341829789898663e00,
            1.0047168154507411e01,
            2.2273672706059052e01,
            5.4049304390828183e01,
            1.5962500533655722e02,
            8.7553100280824765e02,
        ],
    ),
    ((0.05,), (2,)): RationalParam(
        norm_molecular_dynamics=2.6567771557480493e-02,
        residue_molecular_dynamics=[
            5.4415326175599146e-02,
            9.0370308484060427e-02,
            1.8616389644513945e-01,
            4.0655052893921434e-01,
            9.1332229681707799e-01,
            2.2297215526003842e00,
            8.2673806076260998e00,
        ],
        offset_molecular_dynamics=[
            1.1651276172701218e-02,
            3.2128005558694356e-02,
            1.3618256552525199e-01,
            6.4581932161561295e-01,
            3.1710061679824459e00,
            1.6529546460530057e01,
            1.1696719959399913e02,
        ],
        norm_pseudo_fermion=6.6008968113477318e00,
        residue_pseudo_fermion=[
            -4.5244529588728673e-04,
            -2.7913286193527722e-03,
            -1.3910919905953502e-02,
            -6.7151227080820358e-02,
            -3.2392713062715339e-01,
            -1.5980690060213258e00,
            -8.5910485755168793e00,
            -6.3332591104281043e01,
            -1.8771881382968977e03,
        ],
        offset_pseudo_fermion=[
            1.3387699397698993e-02,
            3.1462323899226492e-02,
            9.6415385897133263e-02,
            3.2374992445020234e-01,
            1.1208318145761189e00,
            3.9545250804438625e00,
            1.4540181635176147e01,
            6.1963910644237338e01,
            5.3566592269333353e02,
        ],
        norm_fermion_action=1.5149456635663205e-01,
        residue_fermion_action=[
            6.0458038124269137e-03,
            1.5247824256426594e-02,
            3.7602970968999533e-02,
            9.4912662473817022e-02,
            2.4258769955049089e-01,
            6.3038339460769466e-01,
            1.7214395396814584e00,
            5.5998391201760676e00,
            3.6431278685560251e01,
        ],
        offset_fermion_action=[
            1.1680369733380888e-02,
            2.4528542115263768e-02,
            7.1946920045447768e-02,
            2.3818970133124504e-01,
            8.2029368099576661e-01,
            2.8788453123211895e00,
            1.0425969224178980e01,
            4.1948608522841752e01,
            2.6570653748247554e02,
        ],
    ),
    ((0.0012, 0.0323, 0.2), (2, 1, -3)): RationalParam(
        norm_molecular_dynamics=1.0000021155336281e00,
        residue_molecular_dynamics=[
            1.0939886112519286e-03,
            1.6194518625506564e-03,
            2.9258465733027279e-03,
            5.6669529514102356e-03,
            1.1391958815864864e-02,
            2.6098248081539308e-02,
            2.8374478337435085e-02,
            2.6609540663972843e-02,
            1.5160253740913428e-02,
        ],
        offset_molecular_dynamics=[
            6.4508778820102012e-06,
            1.4125426177598873e-05,
            4.5962277609328359e-05,
            1.7027169639196151e-04,
            6.5239559724009728e-04,
            2.4963044700994116e-03,
            7.8975461377670288e-03,
            3.0287985422753082e-02,
            9.2698483284540570e-02,
        ],
        norm_pseudo_fermion=9.9999990908502512e-01,
        residue_pseudo_fermion=[
            -3.0831906519571126e-08,
            -1.7156350899477111e-07,
            -7.5097306479416274e-07,
            -3.1497834797288229e-06,
            -1.3023111114765444e-05,
            -5.2935793313885423e-05,
            -2.2580671712736666e-04,
            -1.3001913041494567e-03,
            -5.5906615173763850e-03,
            -1.8595160437325720e-02,
            -3.3694177798746595e-02,
        ],
        offset_pseudo_fermion=[
            7.3495561731240404e-06,
            1.5284248344147264e-05,
            4.1035459325352069e-05,
            1.2165520399205419e-04,
            3.7326525096960969e-04,
            1.1603103797459514e-03,
            3.7576263722648704e-03,
            1.1411511818450513e-02,
            3.2394399960427936e-02,
            8.0419448260077409e-02,
            1.4465098374365390e-01,
        ],
        norm_fermion_action=1.0000000909149831e00,
        residue_fermion_action=[
            1.9186889196683772e-05,
            4.5249920190567197e-05,
            1.0243751313969451e-04,
            2.3714605919447510e-04,
            5.5689920043186568e-04,
            1.3342356713808038e-03,
            3.6271108961678806e-03,
            8.2450615844698611e-03,
            1.3751410265182708e-02,
            1.9021897969694522e-02,
            1.2535434676595597e-02,
        ],
        offset_fermion_action=[
            6.5549052650991931e-06,
            1.2318115355807497e-05,
            3.1630328819733195e-05,
            9.2276799534812132e-05,
            2.8157358495104682e-04,
            8.7302072929243489e-04,
            2.7524198040927263e-03,
            7.6125807467459696e-03,
            2.2105508996489113e-02,
            5.8646378877548519e-02,
            1.2246254212695042e-01,
        ],
    ),
    ((0.2,), (1,)): RationalParam(
        norm_molecular_dynamics=1.4922969612472456e-01,
        residue_molecular_dynamics=[
            4.6061009721530329e-02,
            1.1379997711196768e-01,
            2.7453631915038118e-01,
            6.8761853150948216e-01,
            1.8320055253741194e00,
            5.8748098933529755e00,
            3.8086202810075271e01,
        ],
        offset_molecular_dynamics=[
            1.8528257058192529e-01,
            3.7539911712311358e-01,
            1.0581221519179564e00,
            3.4031251236671252e00,
            1.1740502300224540e01,
            4.5730041178720789e01,
            2.8391564624137345e02,
        ],
        norm_pseudo_fermion=2.7327733614742757e00,
        residue_pseudo_fermion=[
            -5.1288622160737063e-03,
            -2.0478919078421993e-02,
            -6.3951352353102550e-02,
            -1.9268600508174474e-01,
            -5.8519459892515535e-01,
            -1.8580289566259904e00,
            -6.6894626972932318e00,
            -3.4401695696851874e01,
            -6.1789841402475429e02,
        ],
        offset_pseudo_fermion=[
            1.8642524371106700e-01,
            3.1582609770026626e-01,
            6.7910597147834129e-01,
            1.6420879353554103e00,
            4.2025399779967945e00,
            1.1215270946596673e01,
            3.2072436285328692e01,
            1.1032630752536383e02,
            7.6441549844451458e02,
        ],
        norm_fermion_action=3.6592862551196720e-01,
        residue_fermion_action=[
            1.0931558584388677e-02,
            2.9290340023195943e-02,
            6.7875566115597738e-02,
            1.5709264288335881e-01,
            3.7002733866738813e-01,
            9.0645109653452494e-01,
            2.4523352073991669e00,
            8.7150280660454786e00,
            7.5897099555179366e01,
        ],
        offset_fermion_action=[
            1.7887536305510454e-01,
            2.9094384593655243e-01,
            6.1203693854712271e-01,
            1.4648617324427462e00,
            3.7284495585739217e00,
            9.8932956134892081e00,
            2.7949316233288556e01,
            9.2734993617230558e01,
            5.4606225005034105e02,
        ],
    ),
    ((0.432,), (1,)): RationalParam(
        norm_molecular_dynamics=1.3593606896652125e-01,
        residue_molecular_dynamics=[
            1.0825758523125259e-01,
            2.3858292199170952e-01,
            4.9757849228595463e-01,
            1.0869288323433812e00,
            2.5985318243095228e00,
            7.8296700630618945e00,
            5.0158518761212449e01,
        ],
        offset_molecular_dynamics=[
            8.2702131812075097e-01,
            1.3808411366467241e00,
            3.0698210123632368e00,
            7.9184834281347021e00,
            2.2574497038355780e01,
            7.5512268302666101e01,
            4.2484648181824633e02,
        ],
        norm_pseudo_fermion=2.8632625463958368e00,
        residue_pseudo_fermion=[
            -1.8589721708993069e-02,
            -6.6618257646310880e-02,
            -1.7948954697327313e-01,
            -4.6191013341188403e-01,
            -1.2063592494878799e00,
            -3.3600738533999448e00,
            -1.0970991346504185e01,
            -5.3397740494988270e01,
            -9.4197553141693140e02,
        ],
        offset_pseudo_fermion=[
            8.3061102769422712e-01,
            1.2147110753467369e00,
            2.1684820046243312e00,
            4.3629198166612913e00,
            9.4367759903301547e00,
            2.1699176279854637e01,
            5.4742751504634327e01,
            1.7132130977818949e02,
            1.1233984291995287e03,
        ],
        norm_fermion_action=3.4925194032896528e-01,
        residue_fermion_action=[
            2.9891626396546980e-02,
            7.3399525035748231e-02,
            1.5049385460469664e-01,
            3.0610458296816817e-01,
            6.3886530145468190e-01,
            1.4125343876406207e00,
            3.5501663088015545e00,
            1.2153536374800286e01,
            1.0508140833810833e02,
        ],
        offset_fermion_action=[
            8.0683696078644196e-01,
            1.1436349138733324e00,
            2.0010625554938715e00,
            3.9795857705315467e00,
            8.5416299143725887e00,
            1.9478232022177817e01,
            4.8385429194643187e01,
            1.4542764755807355e02,
            8.0609481644839570e02,
        ],
    ),
}
