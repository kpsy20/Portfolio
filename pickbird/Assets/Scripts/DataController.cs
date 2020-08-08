using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DataController : MonoBehaviour
{

    private static DataController instance;
    private int ad_num = 1;
    public static DataController GetInstance()
    {
        if (instance == null)
        {
            instance = FindObjectOfType<DataController>();
            if (instance == null)
            {
                GameObject container = new GameObject("DataController");
                instance = container.AddComponent<DataController>();
            }
        }
        return instance;
    }
    
    void Awake()
    {
        ad_num = PlayerPrefs.GetInt("num");
    }
    public void SetNum(int newnum)
    {
        ad_num = newnum;
        PlayerPrefs.SetInt("num", ad_num);
    }
    public void AddNum(int newnum)
    {
        ad_num += newnum;
        SetNum(ad_num);
    }
    public int GetNum()
    {
        return ad_num;
    }

}